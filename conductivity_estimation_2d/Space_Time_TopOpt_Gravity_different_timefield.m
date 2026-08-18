% Function: Space-Time Topology Optimization for Additive Manufacturing: 
%           Concurrent Optimization of Structural Layout and Fabrication Sequence
% Author:  Weiming Wang (wwmdlut@gmail.com)
% Version: 2020-05-14
% Usage:   [xPhys, tPhys] = Space_Time_TopOpt_Gravity(60, 20, 500, 8, 0.5, 0.5);
% volfrac: volume constraint
% nStage:  the number of layers
% nloop:   the number of iterations
% nely:    dimension in y-axis
% nelx:    dimension in x-axis
% Theta:   weight factor alpha in objective function

% [nelx, nely, nloop, nStage, volfrac, Theta] = deal(60,20,500,8,0.5,0.5)
% [xPhys, tPhys, data] = Space_Time_TopOpt_Gravity_different_timefield(90, 30, 300, 12, 0.5, 0.5);
% function [xPhys, tPhys, data] = Space_Time_TopOpt_Gravity_different_timefield(nelx, nely, nloop, nStage, volfrac, Theta)
nelx=180;
nely=60;
nloop=400;
nStage=8;
volfrac=0.5;
Theta=0.1;
tic
fopen('screen.txt','w');
diary screen.txt;
%% Connectivity matrix for continuity 
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
                iH = [iH; e1]; % add element to vector
                jH = [jH; e2];
                sH = [sH; 1];
            end
        end
    end
end
L = sparse(iH,jH,sH); % create sparse(zero's) matrix
M = repmat(sum(L, 2), 1, size(L, 2)); % copies array/matrix of value sum(L,2) to a size 1 x size(L, 2)
E = eye(size(L)); % returns matrix of size L with ones on diagonal and zeros elsewhere
L = E - L./M; % element wise division
L = sparse(L);

%% Material properties
Emax = 1;
Emin = 1e-9;
nu = 0.3;

%% Initialization several parameters
penal = 3;      % stiffness penalty (SIMP method)
rmin = 2;     % density filter radius

%% Stiffness matrix for element
A11 = [12  3 -6 -3;  3 12  3  0; -6  3 12 -3; -3  0 -3 12];
A12 = [-6 -3  0  3; -3 -6 -3 -6;  0 -3 -6  3;  3 -6  3 -6];
B11 = [-4  3 -2  9;  3 -4 -9  4; -2 -9 -4 -3;  9  4 -3 -4];
B12 = [ 2 -3  4 -9; -3  2  9 -2;  4  9  2  3; -9 -2  3  2];
KE = 1/(1-nu^2)/24*([A11 A12;A12' A11]+nu*[B11 B12;B12' B11]);

%% Density filter
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
Hs = sum(H,2); % sum of each row

%% Definition of gravity force
I = []; J = []; S = [];
fe = 1 / (nely*nelx); % ??????
for x = 1 : nely
    for y = 1 : nelx
        I = [I; (y-1)*(nely+1) + x; (y-1)*(nely+1)+x+1; y*(nely+1)+x; y*(nely+1)+x+1];
        J = [J; (y-1)*nely + x; (y-1)*nely + x; (y-1)*nely + x; (y-1)*nely + x];
        S = [S; fe/4; fe/4; fe/4; fe/4];
    end
end
C = sparse(I, J, S);

%%
beta = 1;       % projection parameter
eta = 0.5;      % projection threshold, fixed at 0.5

%% Initialize density field
x = repmat(volfrac,nely,nelx); % nely by nelx array of value volfrac
xTilde = x;
xPhys = (tanh(beta*eta) + tanh(beta*(xTilde-eta))) / (tanh(beta*eta) + tanh(beta*(1-eta))); % step function (continuous)

%% Initialize time field
ypos = linspace(0, nely, nely);
xpos = linspace(0, nelx , nelx);
[xmesh, ymesh] = meshgrid(xpos, ypos);
pos = [xmesh(:) ymesh(:)];
start_pos = [0, 0];
vec = pos - start_pos;
dis2 = sum(vec.*vec, 2);
t = sqrt(dis2) / max(sqrt(dis2));
tPhys = reshape(t,nely, nelx);

figure(2);
imagesc(tPhys);

%% Freedom of degree and loads
F = sparse( 2 * (nelx+1)*(nely+1), 1, -1, 2*(nely+1)*(nelx+1), 1); % Force at the end of the beam, downwards (-1) ?
fixeddofs = 1 : 2*(nely+1);
alldofs = 1 : 2*(nely+1)*(nelx+1);
freedofs = setdiff(alldofs, fixeddofs);

%% Initialization
t = tPhys;
xold1 = x(:);   % transform matrix x into a vector (1 column)
xold2 = x(:);
xold1 = [xold1; zeros(nely*nelx, 1)]; % add zeros for each element
xold2 = [xold2; zeros(nely*nelx, 1)];
low = 0;
upp = 0;
loop = 0;
rou = 10;
% edit 20201214 for iteration plot
objf = zeros(1000,1);   % objective function vector
% consf = zeros(1000,1);
objint = zeros(1000,nStage);    % objective function matrix for each stage ??
vol = zeros(1000,1);

while loop < nloop
    loop = loop+1;
    
    %% Parameter for projection on time field
    if  mod(loop, 30) == 0 && rou < 50  % remainder after division, increase rou every 30 iterations
        rou = rou + 5;
    end
    
    %% Parameter for projection on density field
    if mod(loop, 50) == 0 && beta < 64
        beta = beta * 2;    % increase beta for refinement of density [0,1] (step function) every 50 iterations
    end
    
    %% Objectives and sensitivities
    dc = zeros(nely, nelx);
    dt = zeros(nely, nelx);
    
    %% Compliance of the whole structure
    [c, dcx] = Cal_c_ce_whole(nelx, nely, KE, xPhys, Emin, Emax, penal, freedofs, F);
    obj = c;
    objf(loop) = c;
    dx = beta * (1-tanh(beta*(xTilde-eta)).*tanh(beta*(xTilde-eta))) / (tanh(beta*eta) + tanh(beta*(1-eta)));
    dc(:) = dc(:) + H*(dcx(:).*dx(:)./Hs);
    
    %% Compliances of the intermediate structures
    tP = linspace(0, 1, nStage + 1);
    for i = 1 : nStage
        ti = tP(i+1);
        [c, dcx, dct] = Cal_c_ce_for_gravity(nelx, nely, KE, xPhys, tPhys, ...
            Emin, Emax, penal, ti, C, rou, freedofs);
        objint(loop,i) = c;
        obj = obj + Theta*c;
        dc(:) = dc(:) + Theta*H*(dcx(:).*dx(:)./Hs);
        dt(:) = dt(:) + Theta*H*(dct(:)./Hs);
    end
    
    df0 = [dc, dt];
    df0dx = df0(:);
    f0val = obj;
    n=length(df0dx);
    
    %% Lower and upper bounds for timefield and densityfield
    move = 0.01;    
    xminx=max(0.0, x(:)-move);
    xmaxx=min(1, x(:)+move);
    tmove = 0.01;
    xmint=max(0.0, t(:)-tmove);
    xmaxt=min(1, t(:)+tmove);
    xmin = [xminx; xmint];
    xmax = [xmaxx; xmaxt];
    xval = [x(:); t(:)];
    
    %% Global volume constraint
    fval = sum(sum(xPhys)) / (nelx*nely*volfrac) - 1;
    print_out = fval(end);
    dv = ones(nely,nelx);
    dv(:) = H*(dv(:).*dx(:)./Hs);
    dfdx = [dv(:)'/(nelx*nely*volfrac), zeros(1, nely*nelx)];
    vol(loop) = sum(sum(xPhys))/(nelx*nely);    % sum up elements of density matrix to get volume
    
    %% Continuity constraint
    Nei = 1;   % constrain starting point
    LL = L;
    kk = 2*(nely*nelx); % controlling the smoothness of the time field
    A = LL*tPhys(:);
    B = A.^2/(nely*nelx);
    fval = [fval; kk*(sum(B)-1.0e-6)];
    print_out = [print_out; fval(end)/kk];
    dft = kk*2*LL'*A;
    dft = H*(dft./Hs)/(nely*nelx);
    dfdx = [dfdx; zeros(1, nely*nelx), dft'];
    
    %% Start points
    fval = [fval; (tPhys(Nei)' - 1.0e-9)];
    ss = zeros(length(Nei), nely*nelx);
    for ii = 1 : length(Nei)
        ss(ii, Nei(ii)) = 1;
    end
    dfdx = [dfdx; zeros(length(Nei), nely*nelx), (H*(ss'./repmat(Hs, 1, length(Nei))))'];
    
    %% Volume constraints for intermediate structures
    percent = 1 / nStage;
    tP = linspace(0, 1, nStage+1);
    for i = 1 : nStage
        %%
        ti = tP(i+1);
        ft = 1 - (tanh(rou*ti) + tanh(rou*(tPhys - ti)))/(tanh(rou*ti) + tanh(rou*(1-ti)));
        dfdt = -(rou*(tanh(rou*(tPhys - ti)).^2 - 1))/(tanh(rou*(ti - 1)) - tanh(rou*ti));
        xtJoint = xPhys.*ft;
        fval = [fval; sum(xtJoint(:))/(nelx*nely*volfrac) - i*percent];
        print_out = [print_out; fval(end)];
        dfx = ft/(nelx*nely*volfrac);
        dfx = H*(dfx(:).*dx(:)./Hs);
        dft = xPhys.*dfdt/(nelx*nely*volfrac);
        dft = H*(dft(:)./Hs);
        dfdx = [dfdx; dfx(:)', dft(:)'];
        
        %
        fval = [fval; -sum(xtJoint(:))/(nelx*nely*volfrac) + i*percent - 1.0e-5];
        print_out = [print_out; fval(end)];
        dfdx = [dfdx; -dfx(:)', -dft(:)'];
    end
%     consf(loop) = print_out;
    
    %% Optimizing with MMA solver
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
    
    %% Display and show results
    disp([' It.: ' sprintf('%4i',loop) ' Obj.: ' sprintf('%10.4f',obj) ...
        ' Vol.: ' sprintf('%6.3f',sum(sum(xPhys))/(nelx*nely))]); %' Cons.: ' sprintf('%6.3f',print_out)
    
    if mod(loop, 10) == 0
        figure(1);
        colormap(gray); imagesc(-xPhys, [-1 0]); axis equal; axis tight; axis off; drawnow;
        hold on
        
%         figure(2);
%         colormap(parula);
%         imagesc(tPhys); axis equal; axis tight; axis off; drawnow;
        % % imagesc: The row and column indices of the elements determine the centers of the corresponding pixels
    end
    
    % edit 20201214 to plot the convergence of obj and cons
    % The convergence plot of compliance
%     figure(3)
%     myfontsize = 24;
%     labelsize = 26;
%     plot(1:data.loop,data.objf(1:data.loop),'r-o');
%     hold on
%     xlabel(gca,'Iterations','fontsize',labelsize);
%     ylabel(gca,'Compliance','fontsize',labelsize);
%     set(gca,'FontSize',myfontsize);
%     hold off
%     % The convergence plot of volume fraction
%     figure(4)
%     myfontsize = 24;
%     labelsize = 26;
%     plot(1:data.loop,data.volfrac(1:data.loop),'r-o');
%     hold on
%     xlabel(gca,'Iterations','fontsize',labelsize);
%     ylabel(gca,'volume fraction','fontsize',labelsize);
%     set(gca,'FontSize',myfontsize);
%     hold off
end

% Save data for comparison with robust formulation
%save('xPhys_regular', 'xPhys');
save('tPhys', 'tPhys')


%% edit 20201213 add the combination figure of layout and time field
draw_boundary(tPhys,nStage);
draw_combination(xPhys,tPhys,nStage,1.0e-1);

objf(end)

toc
diary off


% end


%% Subfunctions
%% Calculation of the compliance of the entire structure
function [c,dcx] = Cal_c_ce_whole(nelx, nely, KE, xPhys, Emin, Emax, penal, freedofs, F)

nodenrs = reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx);
edofVec = reshape(2*nodenrs(1:end-1,1:end-1)+1,nelx*nely,1);
edofMat = repmat(edofVec,1,8)+repmat([0 1 2*nely+[2 3 0 1] -2 -1],nelx*nely,1);
iK = reshape(kron(edofMat,ones(8,1))',64*nelx*nely,1);
jK = reshape(kron(edofMat,ones(1,8))',64*nelx*nely,1);
sK = reshape(KE(:)*(Emin+xPhys(:)'.^penal*(Emax-Emin)),64*nelx*nely,1);
K = sparse(iK,jK,sK);
K = (K + K') / 2;  %% do not know why do this, actually K = K'

%%
U = zeros(2*(nely+1)*(nelx+1), 1);
U(freedofs) = K(freedofs, freedofs)\F(freedofs);
ce = zeros(nely, nelx);
ce(1 : nely, 1 : nelx) = reshape(sum((U(edofMat)*KE).*U(edofMat),2), nely, nelx);
c = sum(sum((Emin+xPhys.^penal*(Emax-Emin)).*ce));
dcx = -penal*(Emax-Emin)*xPhys.^(penal-1).*ce;
end

%% Calculation of the compliance of each intermediate structure
function [c, dcx, dct] = Cal_c_ce_for_gravity(nelx, nely, KE, xPhys, tPhys, ...
    Emin, Emax, penal, ti, C, lamda, freedofs)

%% Projection of time field
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
f = -C*xtJoint(:);
F = zeros((nely+1)*(nelx+1), 2);
F(:, 2) = f;
F = F';
F = F(:);

%%
U = zeros(2*(nely+1)*(nelx+1), 1);
U(freedofs) = K(freedofs, freedofs)\F(freedofs);

%%
ce = zeros(nely, nelx);
ce(1 : nely, 1 : nelx) = reshape(sum((U(edofMat)*KE).*U(edofMat),2), nely, nelx);
c = sum(sum((Emin+xtJoint.^penal*(Emax-Emin)).*ce));
dcx1 = -penal*(Emax-Emin)*xtJoint.^(penal-1).*ce.*ft;
dct1 = -penal*(Emax-Emin)*xtJoint.^(penal-1).*ce.*xPhys.*dfdt;
dcx2 = -(U(2:2:end)'*C)'.*ft(:);
dct2 = -(U(2:2:end)'*C)'.*xPhys(:).*dfdt(:);
dcx = 2*dcx2 + dcx1(:);
dct = 2*dct2 + dct1(:);

end

% Publication
% Weiming Wang, Dirk Munro, Charlie C.L. Wang, Fred van keulen, Jun Wu,
% Space-Time Topology Optimization for Additive Manufacturing: 
% Concurrent Optimization of Structural Layout and Fabrication Sequence. 
% Structural and Multidisciplinary Optimization, 61:1-18 (2020).


