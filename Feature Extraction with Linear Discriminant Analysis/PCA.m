function [ V_k ] = PCA( dataset,k )
    [m,n] = size(dataset); 
    % 去除平均值    
    datasetmean = mean(dataset);
    datasetadjust = zeros(m,n);  
    for i = 1 : m   
        datasetadjust(i , :) = dataset(i , :) - datasetmean;  
    end 
    % 计算协方差矩阵
    datacov = cov(datasetadjust);
    % 计算协方差矩阵的特征值与特征向量
    [V, D] = eig(datacov);
    % 将特征值矩阵转换成向量
    d = zeros(1, n);  
    for i = 1:n  
        d(1,i) = D(i,i);  
    end  
    % 对特征值排序
    [maxD, index] = sort(d,'descend');  
    %k个最大的特征值
    index_k = index(1, 1:k);
    % 对应的特征向量 
    V_k = zeros(n,k);  
    for i = 1:k  
        V_k(:,i) = V(:,index_k(1,i));  
    end
end

