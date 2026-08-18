function [ ] = draw_boundary(timing, Ns)
%% Version date 20201214
nely = size(timing, 1);
nelx = size(timing, 2);
mt = timing(:); % time field

% find out the nodal coordinates, not elements center
s = repmat(1 : nely + 1, 1, nelx + 1);
yElement = s(:)-0.5; % according to imagesc
% yElement = s(:)-1;
t = repmat(1 : nelx+1, nely+1, 1);
xElement = t(:)-0.5;
% xElement = t(:)-1;
V = [xElement, yElement];
t = 1:(nely+1)*(nelx+1);
t =reshape(t, nely+1, nelx+1);
t = t(1 : nely, 1 : nelx);
t = t(:);
F = [t, t+nely+1, t+nely+2, t+1]; % clockwise direction, nodal number for each element

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
figure(2)
hold on
for i = 1 : length(Boundary_Edge)
    e = E(Boundary_Edge(i), :);
    draw_line(V(e, :), 3, [0 0 0]);
    hold on
end

% for i= 1:length(Boundary_Edge)
%     edge_nodeA(i)=E(Boundary_Edge(i),1)';
%     edge_nodeB(i)=E(Boundary_Edge(i),2)';
% end
% 
% 
% Edge_node=union(edge_nodeA,edge_nodeB);

end