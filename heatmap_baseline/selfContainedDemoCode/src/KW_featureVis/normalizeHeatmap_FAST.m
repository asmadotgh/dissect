%  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  
%  Normalize Heatmap ( FAST )
%  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  %  

function [normHeatMap] = normalizeHeatmap_FAST( heatmap, minVal, maxVal )

if( sum(heatmap(:))==0 )
    normHeatMap = single( heatmap(:,:,1) ) ;
else    
    isZeroPixel = sum( heatmap, 3 ) == 0 ;
    
    % filter for the positive activations
    heatmap = heatmap .* (heatmap > double(0));

    heatmap =  sum(heatmap.^2,3).^0.5 ;
    heatmap( isZeroPixel ) = 0 ;
    
    if nargin == 1
        normHeatMap = heatmap;
    else

        heatmap = (heatmap - minVal) / (maxVal - minVal);
        normHeatMap = heatmap ;
    end
end


end





