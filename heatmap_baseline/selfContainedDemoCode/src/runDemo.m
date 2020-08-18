 
function [] = runDemo()

%% To avoid displaying figures
%  set(0,'DefaultFigureVisible','off')
%  
%  id = 'MATLAB:prnRenderer:opengl' ;
%  warning('off',id)


%% =====================================================
%% Defining some parameters
%% =====================================================
    
    systemParams.fileDir = fullfile( pwd ) ;
    
    %% Neural Network
    expParams.NN.modelsDir = 'CNNModels' ;
    expParams.NN.imageSetTag = 'ALL' ;

    %% Heatmap Visualization Parameters    
    expParams.NN.heatmap.poolMethodTag = { 'oldPool' ; 'newPool' } ;
    expParams.NN.heatmap.convMethodTag = { 'oldConv' ; 'newConv' } ;

    expParams.NN.heatmap.imageSetTag = 'ALL' ;
    expParams.NN.predHeatmap.imageSetTag = 'VAL' ;
    
    %% Relevant Features
    expParams.SPG.tauVals = [ 10 ] ;
    expParams.SPG.tauEstimationTag = 'IMV-SPAMS';
    
    expParams.visualizations.doGenerateVisualizations = true ;
    expParams.visualizations.showNRelevantFeatures = 3 ;
    
    expParams.fileDir = fullfile( pwd ) ;
    
%%===============================
%% Model to be tested
%%===============================

%% ILSVRC2012
%      datasetTag = 'ILSVRC121' ;

%% ILSVRC2012-cats
    datasetTag = 'ILSVRC12cats1' ;

%%  MNIST
%  datasetTag = 'MNIST1' ;

%% an8-single-6C
%  datasetTag = 'an8FlowerSingleColor' ; 

%% an8-double-12C
%  datasetTag = 'an8FlowerDoubleColor' ;     
    
    
%%===============================    
%% Defining Input Image
%%===============================

%% ILSVRC2012
%      imageName = 'hotairballoon.jpg' ;
%      imageName = 'ILSVRC2012_val_00000278.JPEG' ;
%      imageName = 'ILSVRC2012_val_00000594.JPEG' ;
%      imageName = 'ILSVRC2012_val_00001603.JPEG' ;
%      imageName = 'ILSVRC2012_val_00001912.JPEG' ;
%      imageName = 'ILSVRC2012_val_00002131.JPEG' ;
%      imageName = 'ILSVRC2012_val_00030941.JPEG' ;
       
%% ILSVRC2012-cats
imageName = 'cat_ILSVRC2012_val_00012339.JPEG' ;
%  imageName = 'tiger_copied.png' ;

%%  MNIST
%  imageName = 'MNIST5g_238x238.png' ;
%  imageName = 'MNIST3g_copied.png' ;

%% an8-single-6C
%  imageName = 'class2_img30_p1_r2.jpg' ;
%  imageName = 'class6_img31_p5_r5.jpg' ;

%% an8-double-12C
%  imageName = 'class4_img22_p3_r5.jpg' ;

imageURL = fullfile( pwd,  'testImages', imageName ) ;
    
%%===============================    
%% Defining Heatmap parameters
%%===============================
    
    iConv = 2 ;
    iPool = 1 ;
    cConvMethodTag = expParams.NN.heatmap.convMethodTag{iConv} ;
    cPoolMethodTag = expParams.NN.heatmap.poolMethodTag{iPool} ;
    

%%===============================    
%% Running the actual demo
%%===============================    
    
    computePredictionJustificationHeatmaps( imageURL, datasetTag, cConvMethodTag, cPoolMethodTag, expParams, systemParams ) ;
    disp('processing concluded'); 
    

    
%  %  %      disp('CHECK CURRENT STATUS');
%  %  %      keyboard;              
           
    
    
    
end
