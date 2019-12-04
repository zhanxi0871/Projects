close all
% load data
glassdata = csvread('glassdata.csv')
species = importdata('glassclass.csv');


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
             if isequal(species(k),ClassLabel(3))
                species2(k) = 3;
             else
                 if isequal(species(k),ClassLabel(4))
                    species2(k) = 4;
                 else
                     if isequal(species(k),ClassLabel(5))
                         species2(k) = 5;
                     else
                         species2(k) = 6;
                     end
                 end    
             end
         end
    end
end

% Uniformly choose 30 data as training data
idx = (1:214)';
idx_train = datasample(idx,50,'Replace',false);
idx_test = setdiff(idx,idx_train);
glassdata_train = glassdata(idx_train,:);
species2_train = species2(idx_train);
% Calculate linear discriminant coefficients
s = 3;
[A,mufinal,pi] = PCALDA(glassdata_train,species2_train,s);
%

glassdata_test = glassdata(idx_test,:);
species2_test = species2(idx_test);
 % 
datasetfinal = A * glassdata_test';

 % Classify the training data
[~,f] = size(datasetfinal);
distance = zeros(6,f);
finaldata = zeros(6,f);
final = zeros(1,f);
position = zeros(f,1);
for j = 1:f    
    for m = 1 : 6
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
 
 
 

figure(1)
scatter(finaldata(1,:), finaldata(2,:), 25, species2_test,'filled');
figure(2)
scatter(finaldata(1,:), finaldata(2,:), 25, position,'filled')


% Calculate class probabilities
% P = exp(L) ./ repmat(sum(exp(L),2),[1 2]);