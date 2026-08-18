% Function: Space-Time Topology Optimization for Additive Manufacturing: 
%           Concurrent Optimization of Structural Layout and Fabrication Sequence
% Author:  Weiming Wang (wwmdlut@gmail.com)
% Version: 2020-05-14
% Usage:   [xPhys, tPhys] = Space_Time_TopOpt_Robot(90, 30, 500, 8, 0.6, 0, 0, 0.5);
% volfrac: volume constraint
% nStage:  the number of layers
% nloop:   the number of iterations
% nely:    dimension in y-axis
% nelx:    dimension in x-axis
% Theta:   parameter in objective function
% If Upper_Lower_Print = 1 and Upper_And_Lower_Print = 0, one robot prints up and down from left to right; 
% If Upper_Lower_Print = 0 and Upper_And_Lower_Print = 1, two robots print simultaneously one is up and one is down from left to right;
% If Upper_Lower_Print = 0 and Upper_And_Lower_Print = 0, one robot prints on the top from left to right.
% You cannot choose Upper_And_Lower_Print = 1 and Upper_Lower_Print = 1

% [xPhys, tPhys, data] = Space_Time_TopOpt_Robot(120, 40, 600, 8, 0.6, 0, 0, 0.5);
function [xPhys, tPhys, data] = ...
    Space_Time_TopOpt_Robot(nelx,nely,nloop, nStage, volfrac, ...
    Upper_Lower_Print, Upper_And_Lower_Print, Theta)

close all;
tic
fopen('screen.txt','w');
diary screen.txt;
%% Locations of the robots
tPrint = linspace(0, 1, nStage+1);
xRobot = linspace(1, nelx, nStage + 1)';
yRobot = [zeros(1, length(xRobot)) + 1]';
fPoint = (floor(xRobot)-1)*(nely+1)+yRobot;

if Upper_Lower_Print == 1
    fPointLower = fPoint + nely;
    fPointUpper = fPoint;
    fPoint = [fPointUpper(1:2:end), [fPointLower(2:2:end); 0]];
    fPoint = fPoint';
    fPoint = fPoint(:);
    fPoint(end) = [];
    fPoint(end) = nelx*(nely+1)+1;
    fPoint = unique(fPoint);
    xRobot = floor(xRobot(:));
    yRobot = [yRobot(1:2:end), [yRobot(2:2:end)+nely; zeros(mod(length(xRobot), 2), 1)]]';
    yRobot = yRobot(:);
    yRobot = yRobot(1:length(xRobot), :);
    PT = tPrint(1:end-1);
elseif Upper_And_Lower_Print == 1
    fPointLower = fPoint + nely;
    fPointUpper = fPoint;
    fPoint = [fPointUpper, fPointLower];
    xRobot = [xRobot, xRobot]';
    xRobot = xRobot(:);
    yRobot = [yRobot, yRobot+nely]';
    yRobot = yRobot(:);
    PT = [tPrint(1:end-1)', tPrint(1:end-1)']';
    PT = PT(:);
else
    PT = tPrint(1:end-1);
    fPoint(end) = nelx*(nely+1)+1;
end

rRobot = nely+20; % radius of robot arm
s = repmat(1 : nely, 1, nelx);
yElement = s(:);
t = repmat([1:nelx], nely, 1);
xElement = t(:);
V = [xElement, yElement];

distance = [];
for i = 1 : size(xRobot, 1)
    v = V - repmat([xRobot(i), yRobot(i)], size(V, 1) ,1);
    distance = [distance; sqrt(sum(v.^2, 2))'];
end
eControl = cell(length(PT), 1);
flag = zeros(nely*nelx, length(PT));
for i = 1 : length(PT)
    t = distance(i, :);
    [xt, yt] = find(t < rRobot);
    temp = unique(yt);
    eControl{i} = temp;
    flag(temp, i) = 1;
end

%% Lower and Upper bounds for elements
tUpper = zeros(nely*nelx, 1);
tLower = zeros(nely*nelx, 1);
for i = 1 : size(flag, 1)
    [xx, yy] = find(flag(i, :) == 1);
    tt = PT(yy);
    tm = min(tt); tM = max(tt);
    tLower(i) = tm;
    if Upper_And_Lower_Print
        tUpper(i) = tM+PT(3)-PT(1);
    else
        tUpper(i) = tM+PT(2)-PT(1);
    end
end

%% CONNECTIVITY MATRIX / LAPLACE MATRIX
iH = [];
jH = [];
sH = [];
lrmin = 2;
for i1 = 1:nelx
    for j1 = 1:nely
        e1 = (i1-1)*nely+j1;
        for i2 = max(i1-(ceil(lrmin)-1),1):min(i1+(ceil(lrmin)-1),nelx)
            for j2 = max(j1-(ceil(lrmin)-1),1):min(j1+(ceil(lrmin)-1),nely)
                e2 = (i2-1)*nely+j2;
                if e1 == e2
                    continue;
                end
                iH = [iH; e1];
                jH = [jH; e2];
                sH = [sH; 1];
            end
        end
    end
end
L = sparse(iH,jH,sH);
M = repmat(sum(L, 2), 1, size(L, 2));
E = eye(size(L));
L = E - L./M;
L = sparse(L);

%% MATERIAL PROPERTIES
Emax = 1;
Emin = 1e-9;
nu = 0.3;

%% INITIALIZE SVERAL PARAMETERS
penal = 3;      % stiffness penalty
rmin = 2;     % density filter radius

%% PREPARE FINITE ELEMENT ANALYSIS
A11 = [12  3 -6 -3;  3 12  3  0; -6  3 12 -3; -3  0 -3 12];
A12 = [-6 -3  0  3; -3 -6 -3 -6;  0 -3 -6  3;  3 -6  3 -6];
B11 = [-4  3 -2  9;  3 -4 -9  4; -2 -9 -4 -3;  9  4 -3 -4];
B12 = [ 2 -3  4 -9; -3  2  9 -2;  4  9  2  3; -9 -2  3  2];
KE = 1/(1-nu^2)/24*([A11 A12;A12' A11]+nu*[B11 B12;B12' B11]);

%% PREPARE FILTER DENSITY
iH = ones(nelx*nely*(2*(ceil(rmin)-1)+1)^2,1);
jH = ones(size(iH));
sH = zeros(size(iH));
k = 0;
for i1 = 1:nelx
    for j1 = 1:nely
        e1 = (i1-1)*nely+j1;
        for i2 = max(i1-(ceil(rmin)-1),1):min(i1+(ceil(rmin)-1),nelx)
            for j2 = max(j1-(ceil(rmin)-1),1):min(j1+(ceil(rmin)-1),nely)
                e2 = (i2-1)*nely+j2;
                k = k+1;
                iH(k) = e1;
                jH(k) = e2;
                sH(k) = max(0,rmin-sqrt((i1-i2)^2+(j1-j2)^2));
            end
        end
    end
end
H = sparse(iH,jH,sH);
Hs = sum(H,2);

%% INITIALIZE ITERATION
beta = 1;       % beta continuation
eta = 0.5;      % projection threshold, fixed at 0.5
x = repmat(volfrac,nely,nelx);
xTilde = x;
xPhys = (tanh(beta*eta) + tanh(beta*(xTilde-eta))) / (tanh(beta*eta) + tanh(beta*(1-eta)));

%% Initial time field
s = repmat(1 : nely, 1, nelx);
yElement = s(:)-0.5;
t = repmat([1:nelx], nely, 1);
xElement = t(:)-0.5;
V = [xElement, yElement];
vec = V - repmat(V(1, :), size(V, 1), 1);
dis2 = sum(vec.*vec, 2);
tPhys = sqrt(dis2) / max(sqrt(dis2));
tPhys = reshape(tPhys, nely, nelx);
t = tPhys;

%% Freedom of degree and loads
F = sparse( 2 * (nelx+1)*(nely+1), 1, -1, 2*(nely+1)*(nelx+1), 1);
fixeddofs = 1 : 2*(nely+1);
alldofs = 1 : 2*(nely+1)*(nelx+1);
freedofs = setdiff(alldofs, fixeddofs);

%%
xold1 = reshape(x,[nely*nelx,1]);
xold2 = reshape(x,[nely*nelx,1]);
xold1 = [xold1; tLower];
xold2 = [xold2; tLower];
low = 0;
upp = 0;
color = parula;
rou = 10;
loop = 0;
% edit 20201214 for iteration plot
objf = zeros(1000,1);
% consf = zeros(1000,1);
objint = zeros(1000,nStage-1);
vol = zeros(1000,1);

while loop < nloop
    loop = loop+1;
    
    %% Parameter for projection on time field
    if mod(loop, 30) == 0 && rou < 50
        rou = rou + 5;
    end
    
    %% Parameter for projection on density field
    if mod(loop, 50) == 0 && beta < 64
        beta = beta * 2;
    end
    
    %% OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
    dv = ones(nely,nelx);
    dx = beta * (1-tanh(beta*(xTilde-eta)).*tanh(beta*(xTilde-eta))) / (tanh(beta*eta) + tanh(beta*(1-eta)));
    dv(:) = H*(dv(:).*dx(:)./Hs);
    dc = zeros(nely, nelx);
    dt = zeros(nely, nelx);
    
    %% Objective function
    [c, dcx] = Cal_c_ce_whole(nelx, nely, KE, xPhys, Emin, Emax, penal, freedofs, F);
    obj = c;
    objf(loop) = c;
    dc(:) = dc(:) + H*(dcx(:).*dx(:)./Hs);
    
    tP = linspace(0, 1, nStage + 1);
    for i = 1 : nStage - 1
        fp = [ fPoint(i + 1)];
        force = [-0.5];
        
        if Upper_And_Lower_Print
            fp = [fPoint(i+1, 1); fPoint(i+1, 2)];
            force = [ -0.5;  -0.5];
        end
        
        ti = tP(i+1);
        [c, dcx, dct] = Cal_c_ce_for_weightOfRobot(nelx, nely, KE, xPhys, tPhys, ...
            Emin, Emax, penal, ti, fp, force, rou, freedofs);
        objint(loop,i) = c;
        obj = obj + Theta*c;
        dc(:) = dc(:) + Theta*H*(dcx(:).*dx(:)./Hs);
        dt(:) = dt(:) + Theta*H*(dct(:)./Hs);
    end
    
    df0 = [dc, dt];
    
    %% UPDATE OF DESIGN VARIABLES AND PHYSICAL DENSITIES
    df0dx = df0(:);
    n=length(df0dx);
    move = 0.01;
    tmove = 0.01;
    xmin=max(0.0, x(:)-move);
    xmax=min(1, x(:)+move);
    tmin = max(tLower, t(:)-tmove);
    tmax = min(tUpper, t(:)+tmove);
    xmin = [xmin; tmin];
    xmax = [xmax; tmax];
    xval = [x(:); t(:)];
    f0val = obj;
    
    %% Global volume constraint
    fval = sum(sum(xPhys)) / (nelx*nely*volfrac) - 1;
    dfdx= [dv(:)'/(nelx*nely*volfrac), zeros(1, nely*nelx)];
    vol(loop) = sum(sum(xPhys))/(nelx*nely);
    
    %% Continuity constraint
    LL = L;
    kk = 2*(nely*nelx); % controlling the smoothness of the time field
    A = LL*tPhys(:);
    B = A.^2/(nely*nelx);
    fval = [fval; kk*(sum(B)-1.0e-6)];
    dft = kk*2*LL'*A;
    dft = H*(dft./Hs)/(nely*nelx);
    dfdx = [dfdx; zeros(1, nely*nelx), dft'];
    
    %% Starting point
    fval = [fval; tPhys(1) - 1.0e-9];
    dfdx = [dfdx; zeros(1, nely*nelx), (H*(eye(1, nely*nelx)'./Hs))'];
    
    %% Volumen constraints per layer
    percent = 1/nStage;
    tP = linspace(0, 1, nStage+1);
    for i = 1 : nStage
        %%
        ti = tP(i+1);
        ft = 1 - (tanh(rou*ti) + tanh(rou*(tPhys - ti)))/(tanh(rou*ti) + tanh(rou*(1-ti)));
        dfdt = -(rou*(tanh(rou*(tPhys - ti)).^2 - 1))/(tanh(rou*(ti - 1)) - tanh(rou*ti));
        xtJoint = xPhys.*ft;
        fval = [fval; sum(xtJoint(:))/(nelx*nely*volfrac) - i*percent];
        dfx = ft/(nelx*nely*volfrac);
        dfx = H*(dfx(:).*dx(:)./Hs);
        dft = xPhys.*dfdt/(nelx*nely*volfrac);
        dft = H*(dft(:)./Hs);
        dfdx = [dfdx; dfx(:)', dft(:)'];
        
        %
        fval = [fval; -sum(xtJoint(:))/(nelx*nely*volfrac) + i*percent - 1.0e-5];
        dfdx = [dfdx; -dfx(:)', -dft(:)'];
    end
    
    %% Solving with MMA
    m=length(fval);
    mdof = 1:m;
    a0 = 1;
    a = zeros(m,1);
    c_ = ones(m,1)*1000;
    d = zeros(m,1);
    
    [xmma, ymma, zmma, lam, xsi, eta_, mu, zet, s, low, upp] = ...
        mmasub(m, n, loop, xval, xmin, xmax, xold1, xold2,...
        f0val, df0dx, fval(mdof), dfdx(mdof,:),low, upp, a0, a, c_, d);
    
    xnew = reshape(xmma, nely, []);
    xold2 = xold1;
    xold1 = xval;
    s = xnew(:, 1:nelx);
    xTilde(:) = (H*s(:))./Hs;
    xPhys = (tanh(beta*eta) + tanh(beta*(xTilde-eta))) / (tanh(beta*eta) + tanh(beta*(1-eta)));
    x = s;
    
    %%
    t = xnew(:, nelx+1 : end);
    tPhys = t;
    tPhys(:) =  (H*t(:))./Hs;
        
    %% store results
    objf(loop+1:end) = [];
    objint(loop+1:end,:) = [];
    %     consf(loop+1:end) = [];
    vol(loop+1:end) = [];
    
    data.loop = loop;
    data.objf = objf;
    data.objint = objint;
    %     data.consf = consf;
    data.volfrac = vol;
    
    %% Pring results
    fval(2) = fval(2)/kk;
    disp([' It.: ' sprintf('%4i',loop) ' Obj.: ' sprintf('%10.4f',obj) ...
        ' Vol.: ' sprintf('%6.3f',sum(sum(xPhys))/(nelx*nely)) ...
        ' Cons.: ' sprintf('%6.3f',fval)]);
    
    %% Draw
    if mod(loop, 10) == 0
        figure(1);
        colormap(gray); imagesc(-xPhys, [-1 0]); axis equal; axis tight; axis off; drawnow;
        hold on
        mc = color(10:end-20, :);
        ss = linspace(1, size(mc, 1), nStage+1);
        
        if Upper_Lower_Print
            rx = zeros(length(fPoint), 1);
            rx(2:2:end) = fPoint(2:2:end)/(nely+1) - 0.5;
            rx(1:2:end) = (fPoint(1:2:end) - 1)/(nely+1) + 0.5; ry = yRobot; ry(1:2:end) = ry(1:2:end)-2.2; ry(2:2:end) = ry(2:2:end)-0.5;
        elseif Upper_And_Lower_Print
            rx = zeros(length(fPoint), 2);
            rx(:, 1) = (fPoint(:, 1) - 1)/(nely+1) + 0.5;
            rx(:, 2) = fPoint(:, 2) /(nely+1) - 0.5;
            rx = rx'; rx = rx(:);
            ry = yRobot; ry(1:2:end) = ry(1:2:end)-2.2; ry(2:2:end) = ry(2:2:end)-0.5;
        else
            rx = (fPoint - 1)/(nely+1) + 0.5;
            ry = yRobot-2.2;
        end
        
        for kk = 1:nStage
            if Upper_And_Lower_Print
                quiver(rx(2*kk-1), ry(2*kk-1), 0, 1, 1.5, 'filled', 'linewidth', 4, 'color', mc(round(ss(kk+1)), :));
                hold on;
                quiver(rx(2*kk), ry(2*kk), 0, 1, 1.5, 'filled', 'linewidth', 4, 'color', mc(round(ss(kk+1)), :));
                hold on
            else
                quiver(rx(kk), ry(kk), 0, 1, 1.5, 'filled', 'linewidth', 4, 'color', mc(round(ss(kk+1)), :));
                hold on;
            end
        end
        
        hold on
        quiver(nelx+0.5, nely+0.5, 0, 1, 1.5, 'filled', 'linewidth', 4, 'color', mc(round(ss(kk+1)), :));
%         title('Density Field');
        
        figure(2);
        colormap(parula)
        imagesc(tPhys); axis equal; axis tight; axis off; drawnow;
        hold on
        
%         ss = linspace(1, size(mc, 1), nStage+1);
        for kk = 1:nStage
            if Upper_And_Lower_Print
                quiver(rx(2*kk-1), ry(2*kk-1), 0, 1, 1.5, 'filled', 'linewidth', 4, 'color', mc(round(ss(kk+1)), :));
                hold on;
                quiver(rx(2*kk), ry(2*kk), 0, 1, 1.5, 'filled', 'linewidth', 4, 'color', mc(round(ss(kk+1)), :));
                hold on
            else
                quiver(rx(kk), ry(kk), 0, 1, 1.5, 'filled', 'linewidth', 4, 'color', mc(round(ss(kk+1)), :));
                hold on;
            end
        end
        %
        hold on
        quiver(nelx+0.5, nely+0.5, 0, 1, 1.5, 'filled', 'linewidth', 4, 'color', mc(round(ss(kk+1)), :));
%         title('Timing Field');
    end
    % edit 20201214 to plot the convergence of obj and cons
    % The convergence plot of compliance
    figure(3)
    myfontsize = 24;
    labelsize = 26;
    plot(1:data.loop,data.objf(1:data.loop),'r-o');
    hold on
    xlabel(gca,'Iterations','fontsize',labelsize);
    ylabel(gca,'Compliance','fontsize',labelsize);
    set(gca,'FontSize',myfontsize);
    hold off
    % The convergence plot of volume fraction
    figure(4)
    myfontsize = 24;
    labelsize = 26;
    plot(1:data.loop,data.volfrac(1:data.loop),'r-o');
    hold on
    xlabel(gca,'Iterations','fontsize',labelsize);
    ylabel(gca,'volume fraction','fontsize',labelsize);
    set(gca,'FontSize',myfontsize);
    hold off
end

%% edit 20201213 add the combination figure of layout and time field
draw_boundary(tPhys,nStage);
for kk = 1:nStage
    if Upper_And_Lower_Print
        quiver(rx(2*kk-1), ry(2*kk-1), 0, 1, 1.5, 'filled', 'linewidth', 4, 'color', mc(round(ss(kk+1)), :));
        hold on;
        quiver(rx(2*kk), ry(2*kk), 0, 1, 1.5, 'filled', 'linewidth', 4, 'color', mc(round(ss(kk+1)), :));
        hold on
    else
        quiver(rx(kk), ry(kk), 0, 1, 1.5, 'filled', 'linewidth', 4, 'color', mc(round(ss(kk+1)), :));
        hold on;
    end
end
hold on
quiver(nelx+0.5, nely+0.5, 0, 1, 1.5, 'filled', 'linewidth', 4, 'color', mc(round(ss(kk+1)), :));
% title('Timing Field');

draw_combination(xPhys,tPhys,nStage,1.0e-1);
for kk = 1:nStage
    if Upper_And_Lower_Print
        quiver(rx(2*kk-1)-0.5, -ry(2*kk-1)-1.2, 0, 1, 1.5, 'filled', 'linewidth', 4, 'color', mc(round(ss(kk+1)), :));
        hold on;
        quiver(rx(2*kk)-0.5, -ry(2*kk)-1.2, 0, 1, 1.5, 'filled', 'linewidth', 4, 'color', mc(round(ss(kk+1)), :));
        hold on
    else
        quiver(rx(kk)-0.5, -ry(kk)-1.2, 0, 1, 1.5, 'filled', 'linewidth', 4, 'color', mc(round(ss(kk+1)), :));
        hold on;
    end
end
%
hold on
quiver(nelx, -nely-1.7, 0, 1, 1.5, 'filled', 'linewidth', 4, 'color', mc(round(ss(kk+1)), :));
% title('Timing Field');
        
toc
diary off
end

%%
function [c,dcx] = Cal_c_ce_whole(nelx, nely, KE, xPhys, Emin, Emax, penal, freedofs, F)

nodenrs = reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx);
edofVec = reshape(2*nodenrs(1:end-1,1:end-1)+1,nelx*nely,1);
edofMat = repmat(edofVec,1,8)+repmat([0 1 2*nely+[2 3 0 1] -2 -1],nelx*nely,1);
iK = reshape(kron(edofMat,ones(8,1))',64*nelx*nely,1);
jK = reshape(kron(edofMat,ones(1,8))',64*nelx*nely,1);
sK = reshape(KE(:)*(Emin+xPhys(:)'.^penal*(Emax-Emin)),64*nelx*nely,1);
K = sparse(iK,jK,sK);

%%
U = zeros(2*(nely+1)*(nelx+1), 1);
U(freedofs) = K(freedofs, freedofs)\F(freedofs);

%%
ce = zeros(nely, nelx); 
ce(1 : nely, 1 : nelx) = reshape(sum((U(edofMat)*KE).*U(edofMat),2), nely, nelx);
c = sum(sum((Emin+xPhys.^penal*(Emax-Emin)).*ce));
dcx = -penal*(Emax-Emin)*xPhys.^(penal-1).*ce;

end


%%
function [c, dcx, dct] = Cal_c_ce_for_weightOfRobot(nelx, nely, KE, xPhys, tPhys, ...
                                          Emin, Emax, penal, ti, fpoint, force, lamda, freedofs)

ft = 1 - (tanh(lamda*ti) + tanh(lamda*(tPhys - ti)))/(tanh(lamda*ti) + tanh(lamda*(1-ti)));
dfdt = -(lamda*(tanh(lamda*(tPhys - ti)).^2 - 1))/(tanh(lamda*(ti - 1)) - tanh(lamda*ti));
xtJoint = xPhys.*ft;

%%
nodenrs = reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx);
edofVec = reshape(2*nodenrs(1:end-1,1:end-1)+1,nelx*nely,1);
edofMat = repmat(edofVec,1,8)+repmat([0 1 2*nely+[2 3 0 1] -2 -1],nelx*nely,1);
iK = reshape(kron(edofMat,ones(8,1))',64*nelx*nely,1);
jK = reshape(kron(edofMat,ones(1,8))',64*nelx*nely,1);
sK = reshape(KE(:)*(Emin+xtJoint(:)'.^penal*(Emax-Emin)),64*nelx*nely,1);
K = sparse(iK,jK,sK);
K = (K + K') / 2;
F = zeros((nely+1)*(nelx+1), 2);
F = F'; F = F(:);
F(2*fpoint) = F(2*fpoint)+force;

%%
U = zeros(2*(nely+1)*(nelx+1), 1);
U(freedofs) = K(freedofs, freedofs)\F(freedofs);

ce = zeros(nely, nelx); 
ce(1 : nely, 1 : nelx) = reshape(sum((U(edofMat)*KE).*U(edofMat),2), nely, nelx);
c = sum(sum((Emin+xtJoint.^penal*(Emax-Emin)).*ce));
dcx = -penal*(Emax-Emin)*xtJoint.^(penal-1).*ce.*ft;
dct = -penal*(Emax-Emin)*xtJoint.^(penal-1).*ce.*xPhys.*dfdt;

end

% Publication
% Weiming Wang, Dirk Munro, Charlie C.L. Wang, Fred van keulen, Jun Wu,
% Space-Time Topology Optimization for Additive Manufacturing: 
% Concurrent Optimization of Structural Layout and Fabrication Sequence. 
% Structural and Multidisciplinary Optimization, 61:1-18 (2020).