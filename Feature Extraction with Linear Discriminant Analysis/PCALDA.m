function [ A,mufinal,pi ] = PCALDA( dataset,classdata,s )
    [n,p] = size(dataset);
    c = unique(classdata);
    [k,~] = size(c);
    [b,index1] = sort(classdata);
    class = zeros(k,n);
    e = 0 ;
    r = zeros(k,1);
    for i = 1 : n 
        e = b(i);
        r(e) = r(e) + 1;
        class(e,r(e)) = index1(i);
    end
    n_class = zeros(k,1);
    pi = zeros(k,1);
    mu = zeros(k,p);
    Cov = zeros(p,p);
    for i = 1 : k
        a = [];
        for j = 1 : n
            if class(i,j)>0
                    a = [a class(i,j)]; 
                    n_class(i) = n_class(i) +1;
            end
        end
        pi(i) = n_class(i) / n;
        T = dataset(a,:);
        mu(i,:) = mean(T);
        for y = 1 : n_class(i)
            T_adjust(y,:) = T(y,:) - mu(i,:); 
        end
        for l = 1 : i
            Cov = Cov + T_adjust(l,:)'*T_adjust(l,:);
        end
    end
    sigma = Cov/(n-k);
    [V,D] = eig(sigma)
    d = zeros(1, p);  
    for i = 1:p 
        d(1,i) = D(i,i);  
    end  
    [maxD, index] = sort(d,'descend');  
    index_s = index(1, 1:s);
    d_s = d(1,p-s+1:p);
    D_s = diag(d_s,0);
    V_s = zeros(p,s);  
    for i = 1:s  
        V_s(:,i) = V(:,index_s(1,i));  
    end
    d_trans = zeros(1, s);  
    for i = 1:s  
         d_trans(1,i) = 1/sqrt(D_s(i,i));  
    end  
    D_trans = diag(d_trans,0);
    A = D_trans*V_s';
    mufinal = A * mu';
end

