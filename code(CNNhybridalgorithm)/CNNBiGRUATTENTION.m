%% 导入数据
data = readmatrix(''); % 添加您的文件路径
data = data(:,:);
X = data; % 使用X作为数据变量
% [h1,l1]=data_process(data,10);  
% res = [h1,l1];
% num_samples = size(res,1);   %样本个数

%% 数据预处理
num_samples = size(X,1);       % 样本个数 
kim = 10;                      % 延时步长
zim =  1;                      % 预测时间跨度
or_dim = size(X,2);
outdim = 1;                    % 输出维度

% 重构数据集
res = [];
for i = 1: num_samples - kim - zim + 1
    res(i, :) = [reshape(X(i: i + kim - 1,:), 1, kim*or_dim), X(i + kim + zim - 1,1)]; % 只取第一列作为输出
end

% 训练集和测试集划分
num_size = 0.7;                              % 训练集比例
num_train_s = round(num_size * size(res,1)); % 训练集样本个数
f_ = size(res, 2) - outdim;                  % 输入特征维度

P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

% 归一化处理
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

% 数据平铺
vp_train = double(reshape(p_train, size(p_train,1), 1, 1, size(p_train,2)));
vt_train = t_train;
vp_test = double(reshape(p_test, size(p_test,1), 1, 1, size(p_test,2)));
vt_test = t_test;

%% K-fold交叉验证
k = 10; % 设置折数
cv_indices = crossvalind('Kfold', M, k); % 生成交叉验证索引
cv_mae = zeros(1, k); % 存储每折MAE
cv_rmse = zeros(1, k); % 存储每折RMSE

fprintf('开始%d折交叉验证...\n', k);
for i = 1:k
    fprintf('处理第%d折...\n', i);
    
    % 划分训练/验证集
    val_idx = (cv_indices == i);
    train_idx = ~val_idx;
    
    % 从归一化后的数据中提取
    vp_val_fold = vp_train(:, :, :, val_idx);
    vt_val_fold = vt_train(:, val_idx);
    vp_train_fold = vp_train(:, :, :, train_idx);
    vt_train_fold = vt_train(:, train_idx);
    
    %% 网络搭建CNN-BiGRU-ATTENTION
    numFeatures = size(p_train,1); % 输入特征维度
    
    % 定义自定义翻转层
    flipLayer = functionLayer(@(X) flip(X,2), 'Name', 'flip');
    
    layers = [
        imageInputLayer([numFeatures 1 1], "Name", "sequence")
        
        % CNN部分
        convolution2dLayer([5,1],16,"Name","conv","Padding","same")
        batchNormalizationLayer("Name","batchnorm")
        reluLayer("Name","relu")
        maxPooling2dLayer([2 1],"Name","maxpool","Padding","same")
        flattenLayer("Name","flatten_1")
        fullyConnectedLayer(32,"Name","fc_1")
        
        % BiGRU部分
        gruLayer(10, "Name", "gru1")
        flipLayer % 自定义翻转层
        gruLayer(10, "Name", "gru2")
        
        % Attention和输出
        concatenationLayer(3, 2, "Name", "concat")
        selfAttentionLayer(1, 50, "Name", "selfattention")
        fullyConnectedLayer(outdim, "Name", "fc")
        regressionLayer("Name", "regressionoutput")
    ];
    
    % 连接层
    lgraph = layerGraph(layers);
    lgraph = connectLayers(lgraph, "flatten_1", "concat/in1");
    lgraph = connectLayers(lgraph, "gru1", "concat/in2");
    lgraph = connectLayers(lgraph, "gru2", "concat/in3");

    %% 参数设置
    options = trainingOptions('adam', ...
        'MaxEpochs', 50, ...       % 减少epochs加速交叉验证
        'GradientThreshold', 1, ...
        'InitialLearnRate', 0.01, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', 30, ...
        'LearnRateDropFactor', 0.1, ...
        'L2Regularization', 0.001, ...
        'ExecutionEnvironment', 'cpu',...
        'Verbose', 0, ...           % 关闭详细输出
        'Plots', 'none');

    %% 训练
    net_fold = trainNetwork(vp_train_fold, vt_train_fold', lgraph, options);
    
    %% 预测
    t_sim_val = predict(net_fold, vp_val_fold);
    
    % 反归一化
    T_sim_val = mapminmax('reverse', t_sim_val, ps_output);
    
    % 计算误差
    [mae_val, rmse_val] = calc_error(vt_val_fold', T_sim_val');
    cv_mae(i) = mae_val;
    cv_rmse(i) = rmse_val;
    
    fprintf('折%d - MAE: %.4f, RMSE: %.4f\n', i, mae_val, rmse_val);
end

% 输出交叉验证结果
fprintf('\n%d折交叉验证平均结果:\n', k);
fprintf('平均MAE: %.4f ± %.4f\n', mean(cv_mae), std(cv_mae));
fprintf('平均RMSE: %.4f ± %.4f\n\n', mean(cv_rmse), std(cv_rmse));

%% 使用完整训练集训练最终模型
disp('训练最终模型...');

% 定义自定义翻转层
flipLayer = functionLayer(@(X) flip(X,2), 'Name', 'flip');

% 构建完整网络
numFeatures = size(p_train,1);
layers = [
    imageInputLayer([numFeatures 1 1], "Name", "sequence")
    
    % CNN部分
    convolution2dLayer([5,1],16,"Name","conv","Padding","same")
    batchNormalizationLayer("Name","batchnorm")
    reluLayer("Name","relu")
    maxPooling2dLayer([2 1],"Name","maxpool","Padding","same")
    flattenLayer("Name","flatten_1")
    fullyConnectedLayer(32,"Name","fc_1")
    
    % BiGRU部分
    gruLayer(10, "Name", "gru1")
    flipLayer % 自定义翻转层
    gruLayer(10, "Name", "gru2")
    
    % Attention和输出
    concatenationLayer(3, 2, "Name", "concat")
    selfAttentionLayer(1, 50, "Name", "selfattention")
    fullyConnectedLayer(outdim, "Name", "fc")
    regressionLayer("Name", "regressionoutput")
];

% 连接层
lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph, "flatten_1", "concat/in1");
lgraph = connectLayers(lgraph, "gru1", "concat/in2");
lgraph = connectLayers(lgraph, "gru2", "concat/in3");

%% 参数设置
options = trainingOptions('adam', ...
    'MaxEpochs', 150, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', 0.01, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 60, ...
    'LearnRateDropFactor',0.1, ...
    'L2Regularization', 0.001, ...
    'ExecutionEnvironment', 'cpu',...
    'Verbose', 1, ...
    'Plots', 'training-progress'); % 显示训练进度

%% 训练
tic
net = trainNetwork(vp_train, vt_train', lgraph, options);
toc

%% 预测
t_sim1 = predict(net, vp_train); 
t_sim2 = predict(net, vp_test); 

% 反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
T_train1 = T_train;
T_test2 = T_test;

% 数据格式转换
T_sim1 = double(T_sim1);
T_sim2 = double(T_sim2);

CNN_BiGRU_ATTENTION_TSIM1 = T_sim1';
CNN_BiGRU_ATTENTION_TSIM2 = T_sim2';
save CNN_BiGRU_ATTENTION.mat CNN_BiGRU_ATTENTION_TSIM1 CNN_BiGRU_ATTENTION_TSIM2

%% 指标计算
% 训练集评估
disp('…………训练集误差指标…………')
[mae1,rmse1,mape1,error1] = calc_error(T_train1, T_sim1');
fprintf('MAE: %.4f, RMSE: %.4f, MAPE: %.4f%%\n\n', mae1, rmse1, mape1*100)

% 测试集评估
disp('…………测试集误差指标…………')
[mae2,rmse2,mape2,error2] = calc_error(T_test2, T_sim2');
fprintf('MAE: %.4f, RMSE: %.4f, MAPE: %.4f%%\n\n', mae2, rmse2, mape2*100)

%% 可视化结果
% 训练集结果
figure('Position', [200, 300, 800, 400])
subplot(2,1,1)
plot(T_train1, 'LineWidth', 1.5);
hold on
plot(T_sim1', 'LineWidth', 1.5)
legend('真实值', '预测值')
title('CNN-BiGRU-ATTENTION训练集预测效果对比')
xlabel('样本点')
ylabel('发电功率')
grid on

% 测试集结果
subplot(2,1,2)
plot(T_test2, 'LineWidth', 1.5);
hold on
plot(T_sim2', 'LineWidth', 1.5)
legend('真实值', '预测值')
title('CNN-BiGRU-ATTENTION测试集预测效果对比')
xlabel('样本点')
ylabel('发电功率')
grid on

% 误差曲线
figure('Position', [200, 300, 600, 300])
plot(T_sim2' - T_test2, 'LineWidth', 1.5)
title('CNN-BiGRU-ATTENTION误差曲线图')
xlabel('样本点')
ylabel('发电功率')
grid on

%% 自定义翻转层实现
function flipLayer = functionLayer(func, varargin)
    flipLayer = functionLayer(func, varargin{:});
end