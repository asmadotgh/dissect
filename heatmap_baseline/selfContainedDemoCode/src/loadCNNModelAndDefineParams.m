 
switch( datasetTag )
    
        case 'ILSVRC121' % (~12416 dim)
        
            specsTag = 'imagenet-vgg-f_filterL2L1' 

            % Defining modelURL
            CNNModelURL = fullfile( expParams.NN.modelsDir, 'imagenet-vgg-f.mat' ) ;
            
            % Defining activations (response maps) of interest
            sidx = [ 2:20 ] ;
            activationLevel = 'filterL2L1' ;
        
            net = load( CNNModelURL ) ;
            net = vl_simplenn_tidy( net ) ;
            IMAGE_MEAN = net.meta.normalization.averageImage ;    
            imsize = [224,224];
            IMAGE_MEAN = cat( 3, repmat( IMAGE_MEAN(1) , imsize ), repmat( IMAGE_MEAN(2) , imsize ), repmat( IMAGE_MEAN(3) , imsize ) ) ;   
          
            LAYER_IDX = 21 ;
            IMAGE_DIM = 256;
          
          
        case 'ILSVRC12cats1'

            specsTag = 'ILSVRC12cats_imagenet-vgg-f_filterL2L1' ;
        
            %% Defining modelURL
            CNNModelURL = fullfile( expParams.NN.modelsDir,'ILSVRC12cats_net-epoch-100.mat' ) ;
                        
            %% Defining activations (response maps) of interest
            sidx = [ 2:20 ] ;
            activationLevel = 'filterL2L1' ;
        
            load( CNNModelURL ) ;
            net.layers{1,23}.type = 'softmax';
            selectedLayers = [1:17,19,20,22,23];
            net.layers = net.layers(selectedLayers);
            
            net = vl_simplenn_tidy( net ) ;
            IMAGE_MEAN = net.meta.normalization.averageImage ;    
            imsize = [224,224];
            IMAGE_MEAN = cat( 3, repmat( IMAGE_MEAN(1) , imsize ), repmat( IMAGE_MEAN(2) , imsize ), repmat( IMAGE_MEAN(3) , imsize ) ) ; 
            
            LAYER_IDX = 21 ;        
            IMAGE_DIM = 256;  
          

        case 'MNIST1'
	
            specsTag = 'MNIST-fc_matconvnet_filterL2L1';
	
	    % Defining modelURL
	    CNNModelURL = fullfile( expParams.NN.modelsDir, 'MNIST-fc_net-epoch-20' ) ;
	    	    
	    % Defining activations (response maps) of interest
	    sidx = [ 2:7 ] ;
	    activationLevel = 'filterL2L1' ;
	
	    load( CNNModelURL ) ;
	    net.layers{1,end}.type = 'softmax';
	    
	    net.meta.normalization.imageSize = net.meta.inputSize ;

	    
	    net = vl_simplenn_tidy( net ) ;

	    
	    imsize = [28,28];
	    dummyImageMean = zeros([imsize]) ;
	    IMAGE_MEAN = dummyImageMean ;    
	    
	    
%  	    IMAGE_MEAN = cat( 3,  IMAGE_MEAN, IMAGE_MEAN , IMAGE_MEAN ) ;   
    
            LAYER_IDX = 8 ;	   %% <-- IS THIS NUMBER CORRECT? -- how it was obtained?
	    IMAGE_DIM = 28;

	    
	case 'Fashion144k12c1'    
        
            specsTag = 'Fashion144k_12c_finetuned_vgg-f_filterL2L1' ;
            
            %% Defining modelURL
            CNNModelURL = fullfile( expParams.NN.modelsDir , 'Fashion144k_VGG-F_12classes_1e04_finetuned_net-epoch-300' );

            
            % Defining activations (response maps) of interest
            sidx = [ 2 : 20 ] ;
            activationLevel = 'filterL2L1' ;
        
            net = load( CNNModelURL ) ;
                % Cleaning KW model from training-related layers
                net = net.net;
                net.layers{1,23}.type = 'softmax';
                selectedLayers = [1:17,19,20,22,23];
                net.layers = net.layers( selectedLayers );
                net = vl_simplenn_tidy(net);
            
            %net = vl_simplenn_tidy( net ) ;
            %IMAGE_MEAN = net.meta.normalization.averageImage ;  
            IMAGE_MEAN = net.meta.normalization.averageImage ;    
            imsize = [224,224];
            IMAGE_MEAN = cat( 3, repmat( IMAGE_MEAN(1) , imsize ), repmat( IMAGE_MEAN(2) , imsize ), repmat( IMAGE_MEAN(3) , imsize ) ) ;
            	    
            LAYER_IDX = 21 ;	    
            IMAGE_DIM = 256;	       
            
            
        case 'an8FlowerSingleColor'
        
            
            specsTag = 'an8SynSVFColor_6c_filterL2L1' ;
            
            %% Defining modelURL
            CNNModelURL = fullfile( expParams.NN.modelsDir , 'an8Flower_color_net-epoch-30.mat' );
            
            % Defining activations (response maps) of interest
            sidx = [ 2:9 ] ;
            activationLevel = 'filterL2L1' ;
        
            load( CNNModelURL ) ;
                net.layers{1,end}.type = 'softmax';
                net = vl_simplenn_tidy( net ) ;

            avgImageURL = fullfile( expParams.NN.modelsDir ,'an8Flower_avgImages/an8Flower_color_meanImg.mat' ) ;
            load( avgImageURL ) ;
            IMAGE_MEAN = meanImg ;    
            
            imsize = [ size( meanImg, 1 ), size( meanImg, 1 ) ]   ;
            
            LAYER_IDX = 9 ;
            IMAGE_DIM = 112;
            net.meta.normalization.imageSize = IMAGE_DIM ;
            

        case 'an8FlowerDoubleColor'
        

            specsTag = 'an8SynDVFColor_12c_filterL2L1' ;
            
            %% Defining modelURL
            CNNModelURL = fullfile( expParams.NN.modelsDir , 'an8Flower_doubleColor_net-epoch-30.mat' );
            
            % Defining activations (response maps) of interest
            sidx = [ 2:9 ] ;
            activationLevel = 'filterL2L1' ;
        
            load( CNNModelURL ) ;
                net.layers{1,end}.type = 'softmax';
                net = vl_simplenn_tidy( net ) ;

            avgImageURL = fullfile( expParams.NN.modelsDir ,'an8Flower_avgImages/an8Flower_doubleColor_meanImg.mat' ) ;
            load( avgImageURL ) ;
            IMAGE_MEAN = meanImg ;    
            
            imsize = [ size( meanImg, 1 ), size( meanImg, 1 ) ]   ;
            
            LAYER_IDX = 9 ;
            IMAGE_DIM = 112;
            net.meta.normalization.imageSize = IMAGE_DIM ;
            

        otherwise
            fprintf('UKNOWNN MODEL [%s]... CHECK\n', specsTag );
            keyboard ;
    end
