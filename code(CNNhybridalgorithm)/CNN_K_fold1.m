clc;
clear 
close all


disp('…………………………………………………………………………………………………………………………')
disp('CNN预测模型')
disp('…………………………………………………………………………………………………………………………')

%% 数据预处理
num_samples = length(X);       % 样本个数 
kim = 10;                      % 延时步长
zim =  1;                      % 预测时间跨度
or_dim = size(X,2);

% 重构数据集
res = [];
for i = 1: num_samples - kim - zim + 1
    res(i, :) = [reshape(X(i: i + kim - 1,:), 1, kim*or_dim), X(i + kim + zim - 1,:)];
end

% 训练集和测试集划分
outdim = 1;                                  % 最后一列为输出
num_size = 0.7;                              % 训练集比例
num_train_s = round(num_size * num_samples); % 训练集样本个数
f_ = size(res, 2) - outdim;                  % 输入特征维度

P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

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
    
    P_val_fold = P_train(:, val_idx);
    T_val_fold = T_train(:, val_idx);
    P_train_fold = P_train(:, train_idx);
    T_train_fold = T_train(:, train_idx);
    
    % 归一化处理
    [p_train_fold, ps_input_fold] = mapminmax(P_train_fold, 0, 1);
    p_val_fold = mapminmax('apply', P_val_fold, ps_input_fold);
    
    [t_train_fold, ps_output_fold] = mapminmax(T_train_fold, 0, 1);
    t_val_fold = mapminmax('apply', T_val_fold, ps_output_fold);
    
    % 数据平铺
    trainD_fold = double(reshape(p_train_fold, size(p_train_fold,1), 1, 1, size(p_train_fold,2)));
    valD_fold = double(reshape(p_val_fold, size(p_val_fold,1), 1, 1, size(p_val_fold,2)));
    targetD_fold = t_train_fold;
    
    % 创建CNN网络
    layers = [
        imageInputLayer([size(p_train_fold,1) 1 1], "Name","sequence")
        convolution2dLayer([5,1],16,'Padding','same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer([2 1],'Stride',2)
        fullyConnectedLayer(32)
        reluLayer
        fullyConnectedLayer(1)
        regressionLayer];
   
    % 训练选项
    options = trainingOptions('adam', ...
        'MaxEpochs', 100, ...  % 为加速交叉验证减少轮次
        'GradientThreshold', 1, ...
        'InitialLearnRate', 0.01, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', 70, ...
        'LearnRateDropFactor',0.1, ...
        'L2Regularization', 0.001, ...
        'ExecutionEnvironment', 'cpu',...
        'Verbose', 0, ...       % 关闭详细输出
        'Plots', 'none');
    
    % 训练模型
    net_fold = trainNetwork(trainD_fold, targetD_fold', layers, options);
    
    % 验证集预测
    t_sim_val = predict(net_fold, valD_fold);
    T_sim_val = mapminmax('reverse', t_sim_val, ps_output_fold);
    
    % 计算验证集误差
    [mae_val, rmse_val] = calc_error(T_val_fold, T_sim_val');
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

% 数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

% 数据平铺
trainD = double(reshape(p_train, size(p_train,1), 1, 1, size(p_train,2)));
testD = double(reshape(p_test, size(p_test,1), 1, 1, size(p_test,2)));
targetD = t_train;
targetD_test = t_test;

% 创建CNN网络（与交叉验证相同结构）
layers = [
    imageInputLayer([size(p_train,1) 1 1], "Name","sequence")
    convolution2dLayer([5,1],16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([2 1],'Stride',2)
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer];
   
% 训练选项（使用完整轮次）
options = trainingOptions('adam', ...
    'MaxEpochs', 200, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', 0.01, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 70, ...
    'LearnRateDropFactor',0.1, ...
    'L2Regularization', 0.001, ...
    'ExecutionEnvironment', 'cpu',...
    'Verbose', 1, ...
    'Plots', 'training-progress'); % 显示训练进度

% 训练最终模型
tic
net = trainNetwork(trainD, targetD', layers, options);
toc

% 预测
t_sim1 = predict(net, trainD); 
t_sim2 = predict(net, testD); 

% 反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
T_train1 = T_train;
T_test2 = T_test;

% 格式转换
T_sim1 = double(T_sim1);
T_sim2 = double(T_sim2);

% 保存结果
CNN_TSIM1 = T_sim1';
CNN_TSIM2 = T_sim2';
save CNN.mat CNN_TSIM1 CNN_TSIM2

%% 结果评估
% 训练集评估
disp('…………训练集误差指标…………')
[mae1, rmse1, mape1, error1] = calc_error(T_train1, T_sim1');
fprintf('MAE: %.4f, RMSE: %.4f, MAPE: %.4f%%\n\n', mae1, rmse1, mape1*100)

% 测试集评估
disp('…………测试集误差指标…………')
[mae2, rmse2, mape2, error2] = calc_error(T_test2, T_sim2');
fprintf('MAE: %.4f, RMSE: %.4f, MAPE: %.4f%%\n\n', mae2, rmse2, mape2*100)

%% 可视化结果
% 训练集结果
figure('Position', [200, 300, 800, 400])
subplot(2,1,1)
plot(T_train1, 'LineWidth', 1.5);
hold on
plot(T_sim1', 'LineWidth', 1.5)
legend('真实值', '预测值')
title('CNN训练集预测效果对比')
xlabel('样本点')
ylabel('值')
grid on

% 测试集结果
subplot(2,1,2)
plot(T_test2, 'LineWidth', 1.5);
hold on
plot(T_sim2', 'LineWidth', 1.5)
legend('真实值', '预测值')
title('CNN测试集预测效果对比')
xlabel('样本点')
ylabel('值')
grid on

% 误差曲线
figure('Position', [200, 300, 600, 300])
plot(T_sim2' - T_test2, 'LineWidth', 1.5)
title('CNN预测误差曲线')
xlabel('样本点')
ylabel('预测误差')
grid on