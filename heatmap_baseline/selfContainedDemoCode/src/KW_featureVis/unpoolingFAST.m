function result = unpoolingFAST(feaMapBeforePool_f, feaMapAfterPool_f,  pad, stride, feaMapBackUpLayer,pool )

   numFilter = size(feaMapBeforePool_f,3);
   pool = pool(1);
   if length(pad) == 4
           if pad(1) ~= 0
              paddingRow = pad(1);
              feaMapBeforePool_f(end+1,:,:) = zeros(paddingRow,size(feaMapBeforePool_f,1),numFilter);
           end
           
           if pad(2) ~= 0
              paddingRow = pad(2);
              feaMapBeforePool_f(end+1,:,:) = zeros(paddingRow,size(feaMapBeforePool_f,1), numFilter);
           end
           
           if pad(3) ~= 0
               [l,w,z] = size(feaMapBeforePool_f);
               paddingRow = pad(3);
               feaMapBeforePool_f(:,end+1,:) = zeros(l,paddingRow, numFilter);
           end
           
           if pad(4) ~= 0
               [l,w,z] = size(feaMapBeforePool_f);
               paddingRow = pad(4);
               feaMapBeforePool_f(:,end+1,:) = zeros(l,paddingRow, numFilter);
           end
   else
   end
           
           resultMask = zeros(size(feaMapBeforePool_f));
%            resultMask = gpuArray(resultMask);
           
           x = 1 : stride : size(feaMapBeforePool_f,1);
           y = 1 : stride : size(feaMapBeforePool_f,1);
           
           
           
           Xcoor = length(x) -1;
           Ycoor = length(y) - 1;
           
           if size(feaMapBeforePool_f,1) == 103
               Xcoor = 33;
               Ycoor = 33;
           end
           
           
% % %            for d = 1 : numFilter
% % %            
% % %                 for i = 1 : Xcoor 
% % %                     for j = 1 : Ycoor
% % % 
% % %                         cur_mask_search = feaMapBeforePool_f(y(j) : y(j+1)-1, x(i) : x(i+1)-1,d);
% % %                         cur_mask_pad = resultMask(y(j) : y(j+1)-1, x(i) : x(i+1)-1,d);
% % %                         [a,b] = find(cur_mask_search == feaMapAfterPool_f(j,i,d));
% % %                         
% % %                         if length(a) > 1
% % %                             cur_mask_pad(a(1),b(1)) = feaMapBackUpLayer(j,i,d);
% % %                             
% % %                             resultMask(y(j) : y(j+1)-1, x(i) : x(i+1)-1,d) = cur_mask_pad;
% % %                         else
% % %                             cur_mask_pad(a,b) = feaMapBackUpLayer(j,i,d);
% % %                             
% % %                             resultMask(y(j) : y(j+1)-1, x(i) : x(i+1)-1,d) = cur_mask_pad;                            
% % %                         end
% % %                        
% % %                     end
% % %                 end
% % %                 
% % %            end

           
       

           
           for d = 1 : numFilter
           
                for i = 1 : Xcoor
                    for j = 1 : Ycoor
                        
                        c_value  = feaMapBackUpLayer(j,i,d);
                        
                        
                        cur_mask_search = feaMapBeforePool_f(y(j) : y(j)+pool-1, x(i) : x(i)+pool-1,d);
                        cur_mask_pad = resultMask(y(j) : y(j)+pool-1, x(i) : x(i)+pool-1,d);
                        [a,b] = find(cur_mask_search == max(cur_mask_search(:)));
                        
                        if length(a) > 1
                            cur_mask_pad(a(1),b(1)) = c_value;
                            
                            resultMask(y(j) : y(j)+pool-1, x(i) : x(i)+pool-1,d) =  cur_mask_pad;
                        else
                            cur_mask_pad(a,b) = c_value;
                            
% %                             going_area = resultMask(y(j) : y(j+2)-1, x(i) : x(i+2)-1,d);
                            
                            
                            resultMask(y(j) : y(j)+pool-1, x(i) : x(i)+pool-1,d) =   cur_mask_pad;                            
                        end
                       
                    end
                end
                
           end
          
     if length(pad) == 4
           if pad(1) ~= 0
              resultMask = resultMask((pad(1)):end,:,:);
           end
           
           if pad(2) ~= 0
              
              resultMask = resultMask(1:(end-pad(2)),:,:);
           end
           
           if pad(3) ~= 0
               resultMask = resultMask(:,(pad(3)):end,:);
           end
           
           if pad(4) ~= 0
               resultMask = resultMask(:,1:(end-pad(4)),:);
           end
     else
     end
           
           result = resultMask;
     
           
           


end
