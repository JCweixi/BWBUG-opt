function [itersave,crsave,NFEsave]=SGTO(S,fun_white,fun_black,maxNFE,target,range)

%white surrogate
theta = [1]; lob = [1e-3];upb = [20];
theta = repmat(theta, 1, size(range,2));
lob = repmat(lob, 1, size(range,2));
upb =  repmat(upb, 1, size(range,2));

%initialization
n=size(range,2);
S_SGTO = S;
for i = 1 : size(S_SGTO,1)
    Y_wmodel(i,:) = fun_white(S_SGTO(i,:));
    Y_r(i,:) = fun_black(S_SGTO(i,:));
end

S_white=rlhsamp(1000,n,range);
for i = 1 : size(S_white,1)
    Y_whitemodel(i,:) = fun_white(S_white(i,:));
end
global wmodel;
[wmodel, perf] = dacefit(S_white, Y_whitemodel, @regpoly0,...
        @corrgauss, theta, lob, upb);

Y_rmodel = Y_r;
Y_emodel=Y_rmodel-Y_wmodel;

global emodel;
global black;
[black, perf] = dacefit(S_SGTO, Y_rmodel, @regpoly0,...
        @corrgauss, theta, lob, upb);
[emodel, perf] = dacefit(S_SGTO, Y_emodel, @regpoly0,...
        @corrgauss, theta, lob, upb);

%% optimization
iteration=1;
cr=1e10;
current_number=length(Y_r);

itersave=[1];
crsave=[min(Y_rmodel)];
NFEsave=[size(Y_rmodel,1)];

cr_model=Y_rmodel;

S_div = rlhsamp(10,n,range);
S_resource = repmat(3,1,size(S_div,1));
while current_number<maxNFE && cr>target

    s_temporary = rlhsamp(1000,n,range);
    
    % Black-box primary
    y_black=[];y_bdis=[];
    for i=1:size(s_temporary,1)
        [yb,dz,ybmse,dk]=predictor(s_temporary(i,:),black);
        y_black=[y_black;yb,ybmse];
    end
    
    for i=1:size(s_temporary,1)
        for j=1:size(S_div,1)
            abc=(s_temporary(i,:)-S_div(j,:)).^2;
            y_bdis(i,j)=sqrt(sum(abc(:)));
        end
    end
    
    [y_disv,y_disp] = min(y_bdis');
    
    divnum=cell(1,10);
    for i=1:size(S_div,1)
        divnum{1,i} = find(y_disp==i)';
    end
    
    yb_best=[];
    for i=1:size(S_div,1)
        [yb_sA,yb_index] = sort(y_black(divnum{1,i},1));
        yb_best=[yb_best;s_temporary(divnum{1,i}(yb_index(1:S_resource(i))),:)];
    end
    
    ybmse_best=[];
    for i=1:size(S_div,1)
        [yb_sA,ybmse_index] = sort(y_black(divnum{1,i},2),'descend');
        ybmse_best=[ybmse_best;s_temporary(divnum{1,i}(ybmse_index(1:S_resource(i))),:)];
    end
    
    % White-box review
    y_white=[];
    for i=1:size(s_temporary,1)
        [yw,dz,ywmse,dk]=predictor(s_temporary(i,:),wmodel);
        y_white=[y_white;yw,ywmse];
    end
    
    yw_best=[];
    for i=1:size(S_div,1)
        [yw_sA,yw_index] = sort(y_white(divnum{1,i},1));
        yw_best=[yw_best;s_temporary(divnum{1,i}(yw_index(1)),:)];
    end
    
    ybw_best=[];
    for i=1:size(yw_best,1)
        y_wdis=[];
        for j=1:S_resource(i)
            if i==1
                abc=(yb_best(j,:)-yw_best(i,:)).^2;
            else
                abc=(yb_best(j+sum(S_resource(1:(i-1))),:)-yw_best(i,:)).^2;
            end
            y_wdis(1,j)=sqrt(sum(abc(:)));            
        end
        [y_disv,y_disp] = min(y_wdis');
        if i==1
            ybw_best=[ybw_best;yb_best(y_disp,:)];
        else
            ybw_best=[ybw_best;yb_best(y_disp+sum(S_resource(1:(i-1))),:)];
        end
    end
    
    ybwmse_best=[];
    for i=1:size(yw_best,1)
        y_wmse=[];
        for j=1:S_resource(i)
            if i==1
                y_wmse(1,j)=(predictor(ybmse_best(j,:),wmodel)-predictor(ybmse_best(j,:),black)).^2;  
            else
                y_wmse(1,j)=(predictor(ybmse_best(j+sum(S_resource(1:(i-1))),:),wmodel)-predictor(ybmse_best(j+sum(S_resource(1:(i-1))),:),black)).^2;
            end
        end
        [y_disv,y_disp] = max(y_wmse');
        if i==1
            ybwmse_best=[ybwmse_best;ybmse_best(y_disp,:)];
        else
            ybwmse_best=[ybwmse_best;ybmse_best(y_disp+sum(S_resource(1:(i-1))),:)];
        end
    end
    
   %Grey-box to determine 
    yg_best=[];yg_choose1=[];yg_choose2=[];
    for i=1:size(ybw_best,1)
        yg_choose1(i,:)=predictor(ybw_best(i,:),wmodel);
        yg_choose2(i,:)=predfun_GAS(ybw_best(i,:));
    end  
    [yg_sA1,yg_index1] = sort(yg_choose1);
    [yg_sA2,yg_index2] = sort(yg_choose2);
    
    yg_best=ybw_best(yg_index2(1:2),:);
    yg_index3=find((yg_index1(3:end)-yg_index2(3:end))==0);
    
    if ~isempty(yg_index3) && (corr(Y_wmodel,cr_model)-rand(1,1))>0 
        yg_best=[yg_best;ybw_best(yg_index3(1),:)];
    end
    %
    ygmse_best=[];ygmse_choose1=[];ygmse_choose2=[];
    for i=1:size(ybwmse_best,1)
        ygmse_choose1(i,:)=(predictor(ybwmse_best(i,:),wmodel)-predictor(ybwmse_best(i,:),black)).^2;
        ygmse_choose2(i,:)=(predtest_GAS(ybwmse_best(i,:))-predictor(ybwmse_best(i,:),black)).^2;
    end  
    [ygmse_sA1,ygmse_index1] = sort(ygmse_choose1,'descend');
    [ygmse_sA2,ygmse_index2] = sort(ygmse_choose2,'descend');
    
    ygmse_best=ybwmse_best(ygmse_index2(1:2),:);
    ygmse_index3=find((ygmse_index1(3:end)-ygmse_index2(3:end))==0);
    
    if ~isempty(ygmse_index3) && (corr(Y_wmodel,cr_model)-rand(1,1))>0
        ygmse_best=[ygmse_best;ybwmse_best(ygmse_index3(1),:)];
    end
    
    %Update the model and allocate resources
    points_temp=[yg_best;ygmse_best];
    [S_SGTOS,nnnnnn]=dsmerge([S_SGTO;points_temp],repmat(1,size([S_SGTO;points_temp],1),1),1e-14);
    points=setdiff(S_SGTOS,S_SGTO,'rows');
    
    add_w=[];add_r=[];

    for j = 1 : size(points,1)
        add_w(j,:) = fun_white(points(j,:));
        add_r(j,:) = fun_black(points(j,:));
    end
       
    if ~isempty(add_r)
        Y_r=[Y_r;add_r];
        new_rmodel = Y_r;
        cr_model = new_rmodel;

        S_SGTO=[S_SGTO;points];
        Y_wmodel=[Y_wmodel;add_w];
        Y_emodel=new_rmodel-Y_wmodel;

        [black, perf] = dacefit(S_SGTO, new_rmodel, @regpoly0,...
        @corrgauss, theta, lob, upb);
        [emodel, perf] = dacefit(S_SGTO, Y_emodel, @regpoly0,...
        @corrgauss, theta, lob, upb);
    end
    
    current_number=size(S_SGTO,1);
    iteration=iteration+1;
    [global_value, global_order]=sort(cr_model);
    cr=global_value(1);
    
    itersave=[itersave;iteration];
    crsave=[crsave;cr];
    NFEsave=[NFEsave;current_number];
        
    for i=1:size(points,1)
        for j=1:size(S_div,1)
            if ismember(points(i,:),s_temporary(divnum{1,j},:),'rows')
                resource(1,i)=j;
            end
        end
    end
    
    for i=1:size(S_resource,2)
        if ismember(i,resource)
            S_resource(1,i)=S_resource(1,i)+1;
        else
            S_resource(1,i)=S_resource(1,i)-1;
        end
        if S_resource(1,i)>5
            S_resource(1,i)=5;
        elseif S_resource(1,i)<1
            S_resource(1,i)=1;
        end
    end
      
end
