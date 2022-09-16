clear;
clc;
filename = 'fire8'; OneClassRatio = 0.01;
classifier = strcat('knnClassifier',filename,'.mat');
load(classifier);% knnClassifier; % Got from first frame£¬named 001.jpg

load('fire8AVI.mat'); % selected fire samples
SingleSample = LabelSample(:,1:3);
y = ones(size(SingleSample,1),1); % one-class labels
OneClassModel = fitcsvm(SingleSample,y,'KernelScale','auto','Standardize',true,'OutlierFraction',OneClassRatio);

% loading and reading video
AVIobj = strcat(filename,'.avi');
obj = VideoReader(AVIobj);
f = obj.NumberOfFrames;
Im = read(obj,1);
[row,col,~] = size(Im);
result = ones(f-1,5);
 for j=2:f   
    Im = read(obj,j);
    RGBvec = RGB2vec(Im,row,col);
    P_Label = predict(Mdl,RGBvec);
    P_Label = (P_Label==1);
    GT_Area = 100*sum(P_Label)/size(P_Label,1);
    [TP,FP,Area,T] = predict_OneClass(OneClassModel,RGBvec,P_Label); %One-class SVM 
    result(j-1,:) = [TP,FP,Area,T, GT_Area];   
 end
Frames = (2:1:f)';
j = 1;
figure(j);hold on;
plot(Frames,result(:,j),'rx-.','linewidth',2);
hold on;
box on;grid on;box on;

ylabel('Fire detection rate');
set(gca,'FontWeight','bold','FontSize',13);
set(gcf,'color','w');
axis([-5 250 60 110]);


function [TP,FP,Area,T] = predict_OneClass(OneClassModel,RGBvec,P_Label)
tic;
[~,score] = predict(OneClassModel,RGBvec);
%label = zeros(size(P_Label));
label = (score>=0);
 Area = 100*sum(label)/size(RGBvec,1);
 TP = 100*sum(label(P_Label,:))/sum(P_Label);
 FP = 100*sum(label(~P_Label,:))/sum(~P_Label);
T = toc;
end

function RGBvec = RGB2vec(Im,row,col)
RGBvec = Im(:);
NumPixel = row*col;
RGBvec = reshape(RGBvec,[NumPixel 3]);
RGBvec = double(RGBvec);
end




   
    






