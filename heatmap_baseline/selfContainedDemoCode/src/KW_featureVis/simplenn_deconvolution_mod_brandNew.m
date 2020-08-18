function [heatmap, filterScore] = simplenn_deconvolution_mod_brandNew(net, im_, res, varargin)


% This file is  is made available under the terms of the BSD license (see the LICENCE file)
%
% Parts of the code taken and modified from the FeatureVis library made available under
% the terms of the BSD license (see the LICENCE file).
%
% Parts of the code taken and modified from the MatConvNet library made available
% under the terms of the BSD license (see the MATCONVNET_LICENCE file).
% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.


    % --- process input ---

    % set standard values for optional parameters
    reluMethod = 'guided backpropagation';
    convMethod = 'standard';
    
    if strcmp(net.layers{end}.type, 'softmax')
        measureLayer = length(net.layers)-1;
    else
        measureLayer = length(net.layers);
    end
    measureFilter = 0;

    % parse optional parameters
    for i = 1:2:length(varargin)

        if ~ischar(varargin{i})
            error('The %d. parameter name is not of type char', fix(i/2));
        end

        switch lower(varargin{i})
        case 'relupass'
            if ~ischar(varargin{i+1}) error('The value for ReLUPass must be a char'); end
            reluMethod = lower(varargin{i+1});
        case 'convolutionpass'
            if ~ischar(varargin{i+1}) error('The value for ConvolutionPass must be a char'); end
            convMethod = lower(varargin{i+1});
        case 'measurelayer'
            if ~isnumeric(varargin{i+1}) error('The value for measureLayer must be an integer'); end
            measureLayer = cast(varargin{i+1}, 'int32');
        case 'measurefilter'
            if ~isnumeric(varargin{i+1}) error('The value for measureFilter must be an integer'); end
            measureFilter = cast(varargin{i+1}, 'int32');
        case 'deconvmethod'
            if ~ischar(varargin{i+1}) error('The value for measureFilter must be a char'); end
            deconvMethod = lower(varargin{i+1});
        
        case 'unpoolmethod'
            if ~ischar(varargin{i+1}) error('The value for measureFilter must be a char'); end
            unpoolMethod = lower(varargin{i+1});
            
       
           
                    
        otherwise
            error('Unknown parameter "%s"', varargin{i});
        end
    end

    % check input
    if ~strcmp(reluMethod, 'backpropagation') && ~strcmp(reluMethod, 'deconvnet') && ~strcmp(reluMethod, 'guided backpropagation')
        error('Unknown value for parameter "ReLUPass": "%s"', reluMethod);
    end

    if ~strcmp(convMethod, 'relevance propagation') && ~strcmp(convMethod, 'standard')
        error('Unknown value for parameter "ConvolutionPass": "%s"', convMethod);
    end
    
    if ~strcmp(deconvMethod, 'new') && ~strcmp(deconvMethod, 'old')
        error('Unknow visualization method.');
    end

    if measureLayer < 1
        error('Measure layer (%d) cannot be smaller than 1', measureLayer);
    elseif measureLayer > numel(net.layers)
        error('Measure layer (%d) cannot be greater than number of layers in the network (%d)', measureLayer, numel(net.layers));
    end

    if measureFilter < 0
        error('Measure filter (%d) cannot be negativ', measureFilter);
    end

    % --- preparations ---

    % Result layer one will be the input, so all other layers must be moved up one
    measureLayer = measureLayer + 1;

    gpuMode = isa(im_, 'gpuArray') ;

    % move everything to the GPU
    if gpuMode
        net = vl_simplenn_move(net, 'gpu');
    else
        net = vl_simplenn_move(net, 'cpu');
    end

%  %  %  tic    
%  %  %      % forward pass with dropout disabled
%  %  %      res = vl_simplenn(net, im_, [], [], 'Mode', 'test', 'conserveMemory', false) ;
%  %  %  t1 = toc();
%  t1= 0 ;
    
    % needed to display the classification result of the network
    scores = squeeze(gather(res(end).x)) ;
    [classScore, class] = max(scores) ;

    % if not set by user, set the measure filter to the filter with the
    % maximum activation
    scores = gather(res(measureLayer).x) ;
    
%     filterScore = norm(squeeze((scores(:,:,measureFilter))))/(size(scores,1)*size(scores,2));
    
    if measureFilter == 0 && measureLayer == length(res)
        measureFilter = class;
    elseif measureFilter == 0
        measureFilter = getFilter(scores);
    end

    if measureFilter > size(scores,3)
        error('Measure filter (%d) cannot be greater than the number of filters in layer %d (%d)', ...
            measureFilter, (measureLayer-1), size(scores,3));
    end

    % Display user information
%  %  %      fprintf('Deconvoluting activations of filter %d of layer %d (%s) using %s for the pass through ReLUs and\nthe %s method to pass through the convolutional layers.\n', ...
%  %  %          measureFilter, (measureLayer-1), net.layers{1, (measureLayer-1)}.type, reluMethod, convMethod);

%  tic
    % only keep the filter of interest. set all others to zero
    dzdy = zeros(size(scores,1), size(scores,2), size(scores,3), 'single');
    if gpuMode
        dzdy = gpuArray(dzdy);
    end
    dzdy(:,:,measureFilter) = scores(:,:,measureFilter);
%  t2 = toc();
  
%  tic
    % --- compute deconvolution ---
    res(measureLayer).dzdx = dzdy ;

    res = deconvolution(net, measureLayer, reluMethod, convMethod,deconvMethod,unpoolMethod, res);
%  t3 = toc();
    
    heatmap = gather(res(1).dzdx);
    
%      fprintf('Elapsed time: %.4f | %.4f | %.4f \n', t1, t2, t3);
    
end

function res = deconvolution(net, measureLayer, reluMethod, convMethod,deconvMethod, unpoolMethod, res)

    cudnn = {'CuDNN'} ;

    for i = (measureLayer-1):-1:1

        l = net.layers{i} ;
        res(i).backwardTime = tic ;
        switch l.type
        case 'conv'
            switch deconvMethod
                case 'old'

                    if isfield(l, 'weights')
                        if strcmp(convMethod, 'relevance propagation');
                            eps = 0.001 * ((res(i+1).x > 0) - (res(i+1).x < 0));
                            dzdy = res(i+1).dzdx ./ (res(i+1).x + eps);
                        else
                            dzdy = res(i+1).dzdx;
                        end
                            [dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                            vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                                    dzdy, ...
                                    'pad', l.pad, 'stride', l.stride, ...
                                    cudnn{:}) ;
                        if strcmp(convMethod, 'relevance propagation');
                            res(i).dzdx = dzdx .* res(i).x;
                        else
                            res(i).dzdx = dzdx;
                        end
                    else
                        % Legacy code: will go
                        if strcmp(convMethod, 'relevance propagation');
                            eps = 0.001 * ((res(i+1).x > 0) - (res(i+1).x < 0));
                            dzdy = res(i+1).dzdx ./ (res(i+1).x + eps);
                        else
                            dzdy = res(i+1).dzdx;
                        end
                        [dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                            vl_nnconv(res(i).x, l.filters, l.biases, ...
                                    dzdy, ...
                                    'pad', l.pad, 'stride', l.stride, ...
                                    cudnn{:}) ;
                        if strcmp(convMethod, 'relevance propagation');
                            res(i).dzdx = dzdx .* res(i).x;
                        else
                            res(i).dzdx = dzdx;
                        end
                    end
                    
                case 'new'
                    if isfield(l, 'weights')
                        if strcmp(convMethod, 'relevance propagation');
                            eps = 0.001 * ((res(i+1).x > 0) - (res(i+1).x < 0));
                            dzdy = res(i+1).dzdx ./ (res(i+1).x + eps);
                        else
                            dzdy = res(i+1).dzdx;
                        end
            
                        if l.stride(1) > 1
                
                            stride_ori = l.stride;
                            l.stride = [1,1];
                    
                            convMask = size(net.layers{1,i}.weights{1,1},1);
                            padding = net.layers{1,i}.pad;
                            paddingRows = sum(padding);
try
                            newSizeDzdy = (net.meta.normalization.imageSize(1)+paddingRows - convMask) + 1;
                            resize_dzdy = imresize(dzdy, [newSizeDzdy,newSizeDzdy],'nearest');
catch err
err.message
err.stack(1)
disp('CHECK ERROR');
keyboard;
end

                            [dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                            vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                                resize_dzdy, ...
                                'pad', l.pad, 'stride', l.stride, ...
                                cudnn{:}) ;
                        
                      
                        else
                
                            [dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                            vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                            dzdy, ...
                            'pad', l.pad, 'stride', l.stride, ...
                            cudnn{:}) ;
                        end
                        
                        if strcmp(convMethod, 'relevance propagation');
                            res(i).dzdx = dzdx .* res(i).x;
                        else
                            res(i).dzdx = dzdx;
                        end
                else
            % Legacy code: will go
                    if strcmp(convMethod, 'relevance propagation');
                        eps = 0.001 * ((res(i+1).x > 0) - (res(i+1).x < 0));
                        dzdy = res(i+1).dzdx ./ (res(i+1).x + eps);
                    else
                        dzdy = res(i+1).dzdx;
                    end
                    [dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                        vl_nnconv(res(i).x, l.filters, l.biases, ...
                                dzdy, ...
                                'pad', l.pad, 'stride', l.stride, ...
                                cudnn{:}) ;
                    if strcmp(convMethod, 'relevance propagation');
                        res(i).dzdx = dzdx .* res(i).x;
                    else
                        res(i).dzdx = dzdx;
                    end
             end
            end
        
                    


        case 'convt'
          if isfield(l, 'weights')
            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, ...
                          res(i+1).dzdx, ...
                          'crop', l.crop, 'upsample', l.upsample, ...
                          'numGroups', l.numGroups, cudnn{:}) ;
          else
            % Legacy code: will go
            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnconvt(res(i).x, l.filters, l.biases, ...
                         res(i+1).dzdx, ...
                          'crop', l.crop, 'upsample', l.upsample, ...
                          'numGroups', l.numGroups, cudnn{:}) ;
          end

        case 'pool'
            switch unpoolMethod
                case 'old'
                    res(i).dzdx = vl_nnpool(res(i).x, l.pool, res(i+1).dzdx, ...
                'pad', l.pad, 'stride', l.stride, ...
                'method', l.method, ...
                cudnn{:}) ;
% % %   tic          
% % %                      locations = getMaxPlaces(res(i).x, res(i+1).x, l.pad, l.stride);
% % %             
% % %                      res(i).dzdx = single(unpooling(res(i).x, res(i+1).dzdx, ...
% % %                         l.pad,  l.stride, locations));
% % %   toc
% % %      tic         
                case 'new'
                    res(i).dzdx = single(unpoolingFAST(res(i).x, res(i+1).x, l.pad, l.stride,res(i+1).dzdx,l.pool)); 
            end
% % %      toc       
        case {'normalize', 'lrn'}
            res(i).dzdx = vl_nnnormalize(res(i).x, l.param, res(i+1).dzdx) ;
        case 'softmax'
            res(i).dzdx = vl_nnsoftmax(res(i).x, res(i+1).dzdx) ;
        case 'loss'
            res(i).dzdx = vl_nnloss(res(i).x, l.class, res(i+1).dzdx) ;
        case 'softmaxloss'
            res(i).dzdx = vl_nnsoftmaxloss(res(i).x, l.class, res(i+1).dzdx) ;

        % The behaviour of relus changes depending on the method used
        case 'relu'
            if ~isempty(res(i).x)
                switch reluMethod
                    case 'backpropagation'
                        res(i).dzdx = res(i+1).dzdx .* (res(i).x > single(0)) ;
                    case 'deconvnet'
                        res(i).dzdx = res(i+1).dzdx .* (res(i+1).dzdx > single(0)) ;
                    case 'guided backpropagation'
                        res(i).dzdx = res(i+1).dzdx .* (res(i).x > single(0)) .* (res(i+1).dzdx > single(0)) ;
                end
            else
                % if res(i).x is empty, it has been optimized away, so we use this
                % hack (which works only for ReLU):
                switch reluMethod
                    case 'backpropagation'
                        res(i).dzdx = res(i+1).dzdx .* (res(i+1).x > single(0)) ;
                    case 'deconvnet'
                        res(i).dzdx = res(i+1).dzdx .* (res(i+1).dzdx > single(0)) ;
                    case 'guided backpropagation'
                        res(i).dzdx = res(i+1).dzdx .* (res(i+1).x > single(0)) .* (res(i+1).dzdx > single(0)) ;
                end
            end

        case 'sigmoid'
            res(i).dzdx = vl_nnsigmoid(res(i).x, res(i+1).dzdx) ;
        case 'noffset'
            res(i).dzdx = vl_nnnoffset(res(i).x, l.param, res(i+1).dzdx) ;
        case 'spnorm'
            res(i).dzdx = vl_nnspnorm(res(i).x, l.param, res(i+1).dzdx) ;
        case 'dropout'
            % dropout disabled
            res(i).dzdx = res(i+1).dzdx ;
        case 'bnorm'
            if isfield(l, 'weights')
                [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                    vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, ...
                    res(i+1).dzdx) ;
            else
                [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                    vl_nnbnorm(res(i).x, l.filters, l.biases, ...
                    res(i+1).dzdx) ;
            end
        case 'pdist'
            res(i).dzdx = vl_nnpdist(res(i).x, l.p, res(i+1).dzdx, ...
                'noRoot', l.noRoot, 'epsilon', l.epsilon) ;
        case 'custom'
            res(i) = l.backward(l, res(i), res(i+1)) ;
        end

        res(i).backwardTime = toc(res(i).backwardTime) ;
    end
end

function measureFilter = getFilter(scores)
    % pre-allocate enough space for the activations of all filters
    tempScores = zeros(1, size(scores,3), 'double');

    % go through all filters and calculate the norm of the activation
    for i = 1:size(scores,3)
        tempScores(i) = norm(squeeze(scores(:,:,i)));
    end
    % assign the filter with the maximum activation
    [~, measureFilter] = max(tempScores);
end
