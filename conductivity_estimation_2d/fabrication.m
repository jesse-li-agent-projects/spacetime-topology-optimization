% xPhys=xPhysi;
nely=size(xPhys,1);
nelx=size(xPhys,2);
Xs=zeros(nely,nelx);
XPhys=xPhys(:);
for i=1:length(XPhys(:))
    if XPhys(i)>0.5
        XPhys(i)=1;
    else
        XPhys(i)=0;
    end
end
%%
tPhys1=linspace(0,1,nely*nelx);
% tPhys1=reshape(tPhys1,nelx,nely);
% tPhys=flipud(tPhys1');
tPhys=reshape(tPhys1,nely,nelx);
%%
XPhys=reshape(XPhys,nely,nelx);
xtTS=XPhys.*tPhys;
TPhys=tPhys(:);
xx = find(xtTS(:) == 0);
voidSet = xx;
activeSet = setdiff(1:nely*nelx, voidSet);
[Ts,its]=sort(TPhys);


Its = [];
% Iterate through the bigger array
for i = 1:numel(its)
    % Check if the current element is in the smaller array
    if ismember(its(i), activeSet)
        % Append the element to the new array
        Its = [Its, its(i)];
    end
end
% T_val=1.01-K_est1;
Tval1=T_val.*XPhys(:);
for i=1:nely*nelx
    if Tval1(i)<0.01 && Tval1(i)>0
        Tval1(i)=0.01;
    else
        continue
    end
end
%%
% X1=0.5*XPhys;
v=VideoWriter('exp3dVideoTO5.avi');
open(v);
for i=Its
   Xs(i)=Tval1(i);
   figure(1)
%     colormap(jet); imagesc(Xs); axis equal; axis tight; axis off; drawnow;
% draw_combination1(X1,Xs,0.1);
customColormap = jet(256);
customColormap(1, :) = [1, 1, 1]; % Set color for 0 to white

% Apply the custom colormap
colormap(customColormap);

% Create a sample plot with the custom colormap
imagesc(Xs);

% Display the color limits (can be adjusted as needed)
caxis([0, 1]);
axis equal; axis tight; axis off; drawnow;
    frame=getframe(gcf);
    writeVideo(v,frame);
end

close(v);