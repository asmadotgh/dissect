function [ ] = overlapConfidenceMapOnImage( im, confMap, overlapDirection , outputFileURL, minConf, maxConf)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


%  %  %      addpath('~/Workspace/Matlab/visualizations');

    switch(overlapDirection) 
        case 'im2confMap'
            [h,w,c] = size( confMap ) ;
            im = imresize( im , [h,w] ) ;
        case 'confMap2im' 
            [h,w,c] = size( im ) ;
            confMap = imresize( confMap , [h,w] ) ;
        otherwise
            disp('unknown overlapDirection');
            keyboard;
    end
	



          %building a 3D image from the 1D confMap
            if( nargin < 5 )
                gmap = mat2gray( confMap , [0 1] );
            else
                gmap = mat2gray( confMap , double([ minConf maxConf ]) );
            end
            rgbmap = gray2ind(gmap,256);
            rgbConfMap = ind2rgb(rgbmap,jet(256));
    
            ovim = rgbConfMap ;
        
    
    %overlaping the 2 images
%     figure('Position',[1 h+5 w h])
% %     figure;    
% %     subplottight(1,1,1); imagesc( mat2gray(im) );

    % figureAdv(im, 0.75); imagesc( mat2gray(im) );

    imagesc( mat2gray(im) );
    hold on;
        h = imagesc( rgbConfMap );
        colormap jet;
    hold off;
    set(h,'AlphaData',0.75);
    

    
end


 function subplottight(n,m,i)
     [c,r] = ind2sub([m n], i);
     subplot('Position', [(c-1)/m, 1-(r)/n, 1/m, 1/n]) ;
 end
