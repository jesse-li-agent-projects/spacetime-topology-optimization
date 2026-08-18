function [ ] = draw_combination1(density, timing, eps)
nely = size(density, 1);
nelx = size(density, 2);
Structure = density(:);
[xx, yy] = find(Structure < eps);
Structure(xx) = 0;
Structure = Structure>0;
Structure = reshape(Structure, nely, nelx);  % solid elements
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
F(xx, :) = [];
% % mt(xx) = [];
% time field
mt = timing(:); % time field
mt(xx) = [];

% plot the combination figure of structural layout and time field
figure(1)
% colormap(autumn);
colormap(jet);
patchinfo.Vertices = V;
% patchinfo.Vertices = [xElement-0.5.*ones((nely + 1)*(nelx + 1),1),-yElement+0.5.*ones((nely + 1)*(nelx + 1),1)];
patchinfo.Faces = F(:,1:4);
patchinfo.FaceVertexCData = mt;
patchinfo.FaceColor = 'flat';
patchinfo.EdgeColor = 'none';
patchinfo.LineWidth = 0.5;
patchinfo.CDataMapping = 'scaled';
patch(patchinfo);
% axis off equal
% imagesc(-mt, [-1 0]);
axis equal; axis tight; axis off; drawnow;
%  h=colorbar
% t=get(h,'Limits');
% set(h,'Ticks',linspace(t(1),t(2),5));
hold on
end