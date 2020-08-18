 
function [] = mkdir_if_doesnt_exist( folderURL )

    if( ~exist( folderURL, 'dir' )  )
    
        mkdir( folderURL ) ;
    
    end

end
