close all
% load data
seeddata = csvread('seeddata.csv')
species = importdata('seedclass.csv');


% define species in number 
ClassLabel = unique(species);                               
[n0,~] = size(species);
species2 = zeros(n0,1);
for k = 1:n0
     if isequal(species(k),ClassLabel(1))
        species2(k) = 1;
     else
         if isequal(species(k),ClassLabel(2))
             species2(k) = 2;
         else 
             species2(k) = 3;
         end
    end
end

% Uniformly choose 30 data as training data
idx = (1:151)';
idx_train = datasample(idx,50,'Replace',false);
idx_test = setdiff(idx,idx_train);
seeddata_train = seeddata(idx_train,:);
species2_train = species2(idx_train);
% Calculate linear discriminant coefficients
s = 3;
[A,mufinal,pi] = PCALDA(seeddata_train,species2_train,s);
%

seeddata_test = seeddata(idx_test,:);
species2_test = species2(idx_test);
 % 
datasetfinal = A * seeddata_test';

 % Classify the training data
[~,f] = size(datasetfinal);
distance = zeros(3,f);
finaldata = zeros(3,f);
final = zeros(1,f);
position = zeros(f,1);
for j = 1:f    
    for m = 1 : 3
        Dis = 0;
        for i = 1: s
           distance(i,j) = datasetfinal(i,j) - mufinal(i,m);
           Dis = Dis + distance(i,j)^2;
        end
        finaldata(m,j) = Dis/2-log(pi(m))
    end
    final(j) = min(finaldata(:,j)) 
    position(j)=  find(finaldata(:,j)==final(j))
end
 
 % Calculate the error rate of LDA classifier
 o = find(position ~= species2_test);
 err_rate = length(o)/length(species2_test);
 
 disp('error rate of LDA classifier');
 disp(err_rate);
 
 %% Display Results of LDA
     disp('mean number of error of LDA classifier');     
     disp(mean(k));
     disp('mean error rate of LDA classifier');
     disp(mean(err_rate));
 

figure(1)
scatter(finaldata(1,:), finaldata(2,:), 25, species2_test,'filled');
title('Species of Test Data');
figure(2)
scatter(finaldata(1,:), finaldata(2,:), 25, position,'filled');
title('Predicted Species of Test Data with PCA+LDA')



% Calculate class probabilities
% P = exp(L) ./ repmat(sum(exp(L),2),[1 2]);