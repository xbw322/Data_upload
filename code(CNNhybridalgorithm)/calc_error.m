
function [mae,rmse,mape,error]=calc_error(x1,x2)


error=x2-x1;  %
rmse=sqrt(mean(error.^2));
disp(['MSE:',num2str(mse(x1-x2))])
disp(['RMSE:',num2str(rmse)])

 mae=mean(abs(error));
disp(['MAE:',num2str(mae)])

 mape=mean(abs(error)/x1);
 disp(['MAPE:',num2str(mape*100),'%'])
Rsq1 =( 1 - sum((x1 - x2).^2)/sum((x1 - mean(x2)).^2))*100;
disp(['R2:',num2str(Rsq1),'%'])
end

