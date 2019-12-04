function [ V_k ] = PCA( dataset,k )
    [m,n] = size(dataset); 
    % ȥ��ƽ��ֵ    
    datasetmean = mean(dataset);
    datasetadjust = zeros(m,n);  
    for i = 1 : m   
        datasetadjust(i , :) = dataset(i , :) - datasetmean;  
    end 
    % ����Э�������
    datacov = cov(datasetadjust);
    % ����Э������������ֵ����������
    [V, D] = eig(datacov);
    % ������ֵ����ת��������
    d = zeros(1, n);  
    for i = 1:n  
        d(1,i) = D(i,i);  
    end  
    % ������ֵ����
    [maxD, index] = sort(d,'descend');  
    %k����������ֵ
    index_k = index(1, 1:k);
    % ��Ӧ���������� 
    V_k = zeros(n,k);  
    for i = 1:k  
        V_k(:,i) = V(:,index_k(1,i));  
    end
end

