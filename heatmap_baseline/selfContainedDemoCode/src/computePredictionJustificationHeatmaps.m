 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%	computePredictionJustificationHeatmaps
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
%function [] = computeHeatmapPerPredictedClassFromDataset_paramSet( startImageIdx, endImageIdx, specsTag, datasetInfo, expParams, systemParams )   
function [] = computePredictionJustificationHeatmaps( imageURL, datasetTag, convMethodTag, poolMethodTag, expParams, systemParams ) ;

    addpath( fullfile( systemParams.fileDir, 'src', 'matconvnet') ) ;
    addpath([ systemParams.fileDir , '/src/KW_featureVis' ]) ;

    % Loading network based on specsTag params
    run( [ systemParams.fileDir '/src/matconvnet/matconvnet_matlab/vl_setupnn' ] ) ;
    loadCNNModelAndDefineParams
    
    %% checking the loaded model
    vl_simplenn_display( net ) ;
    disp('check network arch.');
%      keyboard;

    %% Defining additional parameters;
    extraParams.IMAGE_DIM = IMAGE_DIM ; 
    extraParams.IMAGE_MEAN = IMAGE_MEAN ;
    extraParams.sidx = sidx ; 
    extraParams.activationLevel = activationLevel ;
   
    
    %% Loading filter weight Matrix  
    tau = 10 ;
    if( ~isempty( strfind( datasetTag, 'an8' ) ) )
        tau = 5 ;
    end
    inputDir = fullfile( systemParams.fileDir, 'networkActivationsWeights') ;
    inputURL = fullfile( inputDir, sprintf( 'networkActivationWeightMatrix_%s_tau-%d.mat', datasetTag, tau ) ) ;
    %inputURL = fullfile( inputDir, sprintf( 'networkActivationWeightMatrix_%s_tau-%d_FIXED.mat', datasetTag, tau ) ) ;
    load( inputURL, '-mat' ) ;
    
    %% Loading NNActivations Idx    
    inputDir = fullfile( systemParams.fileDir, 'networkActivationsWeights');
    inputFileURL = fullfile( inputDir , ['NNActivationsIdx_' specsTag '.mat'] );
    load( inputFileURL, '-mat' ) ;
    
    

    
    %% =============================
    %% Processing the Image     
    %% =============================
    
    procTime = tic;
    
    fprintf('\t[ %s, %s ] Processing image %s...\n ', convMethodTag, poolMethodTag, imageURL );
                
    %% Computing network response
    [ netResponse , im_ , NNAct ] = computeNetworkResponseFromImage( imageURL, net, extraParams ) ;
    %% Collecting network prediction 
    PRED_LAYER_IDX = LAYER_IDX ;
    scoreOverClasses = netResponse( PRED_LAYER_IDX+1 ).x ;
    [ predScore, predClassId ] = max( scoreOverClasses(:) ) ;

        %% Performing L2-Normalization on the fly
        l2norm = sqrt( sum( NNAct.^2 , 2  ) ) ;
        NNAct = NNAct / l2norm ;
        
        NNAct = double( NNAct ) ;
        
        %% Computing Relevant feature response
        cW = double( W( : , predClassId )' ) ;
            
        feaIntensity = double(NNAct) .* double( cW / sum(cW) ) ;
        feaIdx = find( feaIntensity>0 ) ;
        
        feaIntensity = feaIntensity( feaIdx ) ;
        selFilterLayerInfo = NNActIdx( 2:3, feaIdx  ) ;
    
    %% ===================================
    %% Generating heatmap visualizations
    %% ===================================
    
    combinedHeatmap = double(0) ;
    nRelevantFeatures = size( feaIdx, 2 ) ;
    nRelevantFeatures = min( [ nRelevantFeatures, expParams.visualizations.showNRelevantFeatures ] ) 
    [ sortFeaIntensityVal , sortFeaIntensityIdx ] = sort( feaIntensity , 'descend' ) ;

    relevantFeaturesHeatmaps = zeros([imsize , nRelevantFeatures]) ;
    
    for iFea=1 : nRelevantFeatures
    
        iFeaIdx = sortFeaIntensityIdx(iFea) ;
    
        layerIdx = selFilterLayerInfo( 1, iFeaIdx ) - 1 ;
        filterIdx = selFilterLayerInfo( 2, iFeaIdx ) ;
                
        %% Computing heatmap
        heatmap =  computeHeatmapFromImage( im_ , netResponse , net , convMethodTag, poolMethodTag, layerIdx , filterIdx ) ;
                
        %% Storing every image
        relevantFeaturesHeatmaps(:,:,iFea) = heatmap ;

    end

    %% set each heatmap in the range [0-1]    
    for iFea=1:nRelevantFeatures
        cMap = relevantFeaturesHeatmaps(:,:,iFea) ;
        minVal = min( cMap(:) ) ;
        maxVal = max( cMap(:) ) ;
        relevantFeaturesHeatmaps(:,:,iFea) = ( relevantFeaturesHeatmaps(:,:,iFea) - minVal ) / ( maxVal - minVal ) ;
    end    
    
    %% Computing combined heatmap
    %% sorting test image activations responses
    combinedHeatmap = max( relevantFeaturesHeatmaps, [],  3 ) ;    
    
    %% =====================================
    %% Displaying / Storing visualizations
    %% =====================================
    
    %% for display purposes
     if( ~isempty( strfind( datasetTag, 'MNIST' ) ) )
        disp('MNIST dataset detected');
        predClassId = predClassId - 1 ; % since MNIST classes are listed from 0 to 9
    end
    
    %% Defining directory to store the computed heatmaps
    outputDir = fullfile( expParams.fileDir, 'output', datasetTag );
    mkdir_if_doesnt_exist( outputDir ) ;

    fileType = 'relevantFeaturesHeatmaps' ;
    fileExt = 'mat' ;
    
    outputURL = fullfile( outputDir, sprintf( 'relevantFeaturesHeatmaps_predClass-%d.mat', predClassId  ) ) ;    
    save( outputURL , 'relevantFeaturesHeatmaps'  , '-mat' ) ;    
    
    %% Defining directory to store the computed visualizations
    if( expParams.visualizations.doGenerateVisualizations )
        
        
        overlapDirection = 'confMap2im';
        colormap 'jet';
    
        outputDir = fullfile( expParams.fileDir, 'visualizations', datasetTag );
        mkdir_if_doesnt_exist( outputDir ) ;
        
        fileNamePattern = '%s_%s_predClass-%d_rank-%d_l-%d_f-%d.%s';

        
        fileType = 'heatmapVisualization' ;
        fileExt = 'png' ;
            
        im = imread( imageURL )     ;
        [~,imageName,~] = fileparts( imageURL ) ;   
        

        for iFea=1 : nRelevantFeatures
        
            iFeaIdx = sortFeaIntensityIdx(iFea) ;

            layerIdx = selFilterLayerInfo( 1, iFeaIdx ) - 1 ;
            filterIdx = selFilterLayerInfo( 2, iFeaIdx ) ;
            
            visURL = fullfile( outputDir, sprintf( fileNamePattern, imageName, fileType, predClassId, iFea, layerIdx, filterIdx, fileExt  ) ) ;
            
            confMap = relevantFeaturesHeatmaps(:,:,iFea) ;
            
            figure;    
            overlapConfidenceMapOnImage( im, confMap, overlapDirection );
            
            colormap jet;
            colorbar;
            title( sprintf('pred. class: %d (rank-%d layer-%d filter-%d)', predClassId, iFea, layerIdx, filterIdx ) ) ;
            export_fig( visURL ) ;
            
            
        end
        
        %% Showing combined heatmap
        figure; 
        overlapConfidenceMapOnImage( im, combinedHeatmap, overlapDirection );
        
        title('combined heatmap');
        colormap jet;
        colorbar;
        
        visURL = fullfile( outputDir, sprintf( '%s_combinedHeatmapVisualization_predClass-%d.%s', imageName, predClassId, fileExt  ) ) ;
        export_fig( visURL ) ;
        
        
    end
    
    
    fprintf('\n done... (%.2f secs.)' , toc(procTime) );
    

                
end




%%  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  
%%   computeHeatmapFromImage  
%%  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  



function [ normHeatmap ] = computeHeatmapFromImage( im_ , networkResponse , net , convMethodTag, poolMethodTag, layerIdx , filterIdx  ) 

    %% Computing heatmap
    if( strcmpi( convMethodTag, 'oldConv' ) && strcmpi( poolMethodTag, 'oldPool' ) )
        heatmap = simplenn_deconvolution_mod_brandNew( net, im_, networkResponse,'MeasureLayer', layerIdx, 'MeasureFilter', filterIdx, 'deconvMethod','old', 'unpoolMethod','old') ;
        
    elseif( strcmpi( convMethodTag, 'oldConv' ) && strcmpi( poolMethodTag, 'newPool' ) )
	heatmap = simplenn_deconvolution_mod_brandNew( net, im_, networkResponse,'MeasureLayer', layerIdx, 'MeasureFilter', filterIdx, 'deconvMethod','old', 'unpoolMethod','new') ;
    
    elseif( strcmpi( convMethodTag, 'newConv' ) && strcmpi( poolMethodTag, 'oldPool' ) )
        net.normalization = net.meta.normalization ; 
        heatmap = simplenn_deconvolution_mod_brandNew( net, im_, networkResponse,'MeasureLayer', layerIdx, 'MeasureFilter', filterIdx, 'deconvMethod','new', 'unpoolMethod','old') ;

    elseif( strcmpi( convMethodTag, 'newConv' ) && strcmpi( poolMethodTag, 'newPool' ) )
        heatmap = simplenn_deconvolution_mod_brandNew( net, im_, networkResponse,'MeasureLayer', layerIdx, 'MeasureFilter', filterIdx, 'deconvMethod','new', 'unpoolMethod','new') ;
    
    end
     
     
    %% Normalizing heatmap intensity
%  %  %      normHeatmap = normalizeHeatmap_FAST( heatmap );
    normHeatmap = normalizeHeatmap_FAST( heatmap, min(heatmap(:)), max(heatmap(:)) );

end


%%  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  
%%   computeNetworkResponseFromImage  
%%  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  

% Data coming in from matlab needs to be in the order
%   [width, height, channels, images]
% where width is the fastest dimension.
% Here is the rough matlab for putting image data into the correct
% format in W x H x C with BGR channels:
%   % permute channels from RGB to BGR
%   im_data = im(:, :, [3, 2, 1]);
%   % flip width and height to make width the fastest dimension
%   im_data = permute(im_data, [2, 1, 3]);
%   % convert from uint8 to single
%   im_data = single(im_data);
%   % reshape to a fixed size (e.g., 227x227).
%   im_data = imresize(im_data, [IMAGE_DIM IMAGE_DIM], 'bilinear');
%   % subtract mean_data (already in W x H x C with BGR channels)
%   im_data = im_data - mean_data;

function [netResponse , im_ , CNNAct ] = computeNetworkResponseFromImage( imageURL, net,  extraParams )
        IMAGE_MEAN = extraParams.IMAGE_MEAN  ;
        IMAGE_DIM = extraParams.IMAGE_DIM ;
        CROPPED_DIM = net.meta.normalization. imageSize(1) ;
        sidx = extraParams.sidx ;
        activationLevel = extraParams.activationLevel ;
        
	try
	
            try    
                im = imread( imageURL ) ;
            catch err
                if( strcmpi( err.identifier, 'MATLAB:imagesci:jpg:cmykColorSpace' ) )
                    imageURL = extraParams.fixImageURL ;
                    im = imread( imageURL ) ;
                end
            end    
            
            im_ = single( im ) ; % note: 0-255 range
            im_ = imresize( im_, [IMAGE_DIM, IMAGE_DIM] ) ;
            

            im_ = imresize( im_, [CROPPED_DIM, CROPPED_DIM] ) ;
            im_ = bsxfun( @minus, im_, IMAGE_MEAN ) ;

            netResponse = vl_simplenn( net, im_ ) ; 

            CNNAct = collectNetworkActivationsOfInterest( netResponse, sidx, activationLevel ) ;
            
	
        catch err
	    disp('error while loading image');
	    imageURL
	    keyboard;
	end	

end




%%  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  
%%  collectNetworkActivationsOfInterest
%%  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  


function [ activations ] = collectNetworkActivationsOfInterest( res, sidx, activationLevel )

    mSelectedMaps = length( sidx ) ;
    
    activations = [] ;
    for iMap=1 : mSelectedMaps
        cidx = sidx(iMap) ;

        switch( activationLevel )
        
            case 'raw'
            
                % all activations at all position in the response map
                %disp( [ 'Layer-' int2str(cidx) ' : '  int2str( size( res(cidx).x ) ) ] ) ;
                activations = [ activations , res( cidx ).x(:)' ] ;
            
            case 'filterL2L1'
            
                % filter-wise response (squared-L2 norm of each channel)
                l2norm = res( cidx ).x .^ 2 ; %% raising every element of every channel to the power 2
                l2norm = sum( l2norm , 1 ) ;  %% adding everything within one dimension
                l2norm = sum( l2norm , 2 ) ;  %% adding everything within the other dimension              
                l2norm = l2norm(:) ;
                
                % L1-normalization of the filter-wise responses
                cActivations = l2norm ./ sum(l2norm) ;
                % concatenating activations
                activations = [ activations, cActivations(:)' ] ;
            
            
            otherwise
            
                disp('ERROR at collectNetworkActivationsOfInterest');
                keyboard;
        end
        
    end

end
