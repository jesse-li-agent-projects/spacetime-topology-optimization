function [ ] = draw_combination3(XPhys, tPhys, T1)
%% Version date 20201213

% controal parameter
% eps: Trunction value for identifying solid elements

% structural layout, density field
timing=tPhys;
density=XPhys;
temp=T1;
Ns=8;
j=8;
eps=0.1;


mt = timing(:); % time field
tt = linspace(0, 1, Ns+1);
 for j = 1 :8
    [xx1, yy1] = find(mt <= tt(j+1));
    if j == 1
        [xx2, yy2] = find(mt >= tt(j));
    else
        [xx2, yy2] = find(mt > tt(j));
    end
    
    xx = intersect(xx1, xx2);
    mt(xx) = j+1;
 end

nely = size(density, 1);
nelx = size(density, 2);
Structure = density(:);
Structure(mt<1)=0;
[xx]=find(Structure<=0);
Temperature=temp(:);
Temperature(mt<1)=0;
Structure = reshape(Structure, nely, nelx);  % solid elements
Temperature = reshape(Temperature, nely, nelx);
% % md = Structure(:); 

% find out the center coordinates of elements
s = repmat(1 : nely + 1, 1, nelx + 1);
% yElement = s(:)-0.5;
yElement = s(:)-1;
t = repmat([1:nelx+1], nely+1, 1);
% xElement = t(:)-0.5;
xElement = t(:)-1;
% find out elements' nodal numbering
V = [xElement, -yElement];
t = 1:(nely+1)*(nelx+1);
t =reshape(t, nely+1, nelx+1);
t = t(1 : nely, 1 : nelx);
t = t(:);
F = [t, t+nely+1, t+nely+2, t+1]; % clockwise direction, code number for elements
% % md=Structure(:);
% % [xx, yy]=find(md == 0);
%  F(xx, :) = [];
% % mt(xx) = [];
% time field
mt1=zeros(nely,nelx);
for i=1:nely*nelx
    if sum(ismember(xx,i))>0
           continue
    else
        mt1(i) = Temperature(i);

    end
end
mt1=mt1(:);
% mt1(xx) = [];


% plot the combination figure of structural layout and time field
figure(1)
% colormap(autumn);
colormap(jet);
patchinfo.Vertices = V;
% patchinfo.Vertices = [xElement-0.5.*ones((nely + 1)*(nelx + 1),1),-yElement+0.5.*ones((nely + 1)*(nelx + 1),1)];
patchinfo.Faces = F(:,1:4);
patchinfo.FaceVertexCData = mt1;
patchinfo.FaceColor = 'flat';
patchinfo.EdgeColor = 'none';
patchinfo.LineWidth = 0.5;
patchinfo.CDataMapping = 'scaled';
patch(patchinfo);
% axis off equal
% imagesc(-mt, [-1 0]);
colorbar; axis equal; axis tight; axis off; drawnow;
  h=colorbar
t=get(h,'Limits');
set(h,'Ticks',linspace(t(1),t(2),5));
myColorMap = jet(256);
myColorMap(1,:) = 1;
colormap(myColorMap);
hold on
% end

%% The below part is adopted from Weiming, which plot the boundary lines between each layer
%% This part be used after holding on the combination figure of structural layout and time field
% find out which stage does each element belong to
tt = linspace(0, 1, Ns+1);
for j = 1 : Ns
    [xx1, yy1] = find(mt <= tt(j+1));
    if j == 1
        [xx2, yy2] = find(mt >= tt(j));
    else
        [xx2, yy2] = find(mt > tt(j));
    end
    
    xx = intersect(xx1, xx2);
    mt(xx) = j+1;
end

%
Edge_Face = sparse(size(V, 1), size(V, 1));
E = [F(:, 1), F(:, 2); F(:, 2) F(:, 3); F(:, 3) F(:, 4); F(:, 4) F(:, 1)];
E = sort(E, 2);
E = unique(E, 'rows');  %get all neighbouring node pairs
for i = 1 : size(F, 1)
    e = [F(i, :)', circshift(F(i, :), -1)'];
    for j = 1 : size(e, 1)
        Edge_Face(e(j, 1), e(j, 2)) = i; %element number is appointed to its four edges
    end
end
Boundary_Edge = [];

for i = 1 : size(E, 1)
    f1 = Edge_Face(E(i, 1), E(i, 2));
    f2 = Edge_Face(E(i, 2), E(i, 1));
    if f1 == 0 || f2 == 0
        continue;
    else
        if mt(f1) ~= mt(f2)
            Boundary_Edge = [Boundary_Edge; i];   %find out all the element edges where variation happens
        end
    end
end

% hold on
for i = 1 : length(Boundary_Edge)
    e = E(Boundary_Edge(i), :);
    draw_line(V(e, :), 3, [0 0 0]);
    hold on
end
% % hold on for time field picture
% figure(2)
% for i = 1 : length(Boundary_Edge)
%     e = E(Boundary_Edge(i), :);
%     draw_line(V(e, :), 3, [0 0 0]);
%     hold on
% end
end