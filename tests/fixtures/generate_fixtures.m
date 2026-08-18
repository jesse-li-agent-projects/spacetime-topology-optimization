%% Fixture generation harness for the Python port of conductivity_estimation_2d.
%
% Regenerates every .mat file under tests/fixtures/ from a small (nelx ~= nely),
% asymmetric, fixed-seed problem instance, driven by the *actual* MATLAB source:
% mmasub.m/subsolv.m are called unmodified, on path. Cal_c_ce_whole and
% Cal_c_ce_for_gravity are duplicated verbatim as local functions below (MATLAB
% can't call another script's local functions from outside that script) --
% their bodies are copy-pasted from conductivity_estimation_stto_main.m, not
% reimplemented, so they remain the reference ground truth.
%
% See sttopt/conventions.md for the array-order and fixture-format conventions
% this script follows (nelx != nely, COO triplets for neighbor lists, plain
% numeric arrays with an explicit iteration dimension instead of struct arrays).
%
% Run from the repo root:
%   matlab -nodisplay -nosplash -batch "run('tests/fixtures/generate_fixtures.m')"

clear; close all;
addpath(fullfile(pwd, 'conductivity_estimation_2d'));
out = fullfile(pwd, 'tests', 'fixtures');

%% Problem size -- deliberately small, asymmetric (nelx != nely), fixed-seed.
nelx = 7;
nely = 5;
nStage = 3;
volfrac = 0.5;
Theta = 0.1;
Tcr = 0.8;
tfield = 3;
nloop = 3;
beta = 1;
eta = 0.5;
factor = 1;

%% Continuity filter L
lrmin = 2;
iH = []; jH = []; sH = [];
for i1 = 1:nelx
    for j1 = 1:nely
        e1 = (i1-1)*nely+j1;
        for i2 = max(i1-(ceil(lrmin)-1),1):min(i1+(ceil(lrmin)-1),nelx)
            for j2 = max(j1-(ceil(lrmin)-1),1):min(j1+(ceil(lrmin)-1),nely)
                e2 = (i2-1)*nely+j2;
                if e1 == e2
                    continue;
                end
                iH = [iH; e1]; jH = [jH; e2]; sH = [sH; 1];
            end
        end
    end
end
L = sparse(iH,jH,sH);
M = repmat(sum(L, 2), 1, size(L, 2));
E = eye(size(L));
L = E - L./M;
L = sparse(L);

%% Material properties / element stiffness
Emax = 1; Emin = 1e-9; nu = 0.3; penal = 3;
A11 = [12  3 -6 -3;  3 12  3  0; -6  3 12 -3; -3  0 -3 12];
A12 = [-6 -3  0  3; -3 -6 -3 -6;  0 -3 -6  3;  3 -6  3 -6];
B11 = [-4  3 -2  9;  3 -4 -9  4; -2 -9 -4 -3;  9  4 -3 -4];
B12 = [ 2 -3  4 -9; -3  2  9 -2;  4  9  2  3; -9 -2  3  2];
KE = 1/(1-nu^2)/24*([A11 A12;A12' A11]+nu*[B11 B12;B12' B11]);

% edofMat is computed inline inside Cal_c_ce_whole/Cal_c_ce_for_gravity in the
% original; recomputed identically here so Phase 1 (fem.py) has a standalone
% fixture for it.
nodenrs = reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx);
edofVec = reshape(2*nodenrs(1:end-1,1:end-1)+1,nelx*nely,1);
edofMat = repmat(edofVec,1,8)+repmat([0 1 2*nely+[2 3 0 1] -2 -1],nelx*nely,1);

save(fullfile(out, 'fem_setup.mat'), 'KE', 'edofMat', 'nelx', 'nely', '-v7');

%% Density filter H, Hs
rmin = 2;
iH1 = ones(nelx*nely*(2*(ceil(rmin)-1)+1)^2,1);
jH1 = ones(size(iH1)); sH1 = zeros(size(iH1));
k = 0;
for i1 = 1:nelx
    for j1 = 1:nely
        e1 = (i1-1)*nely+j1;
        for i2 = max(i1-(ceil(rmin)-1),1):min(i1+(ceil(rmin)-1),nelx)
            for j2 = max(j1-(ceil(rmin)-1),1):min(j1+(ceil(rmin)-1),nely)
                e2 = (i2-1)*nely+j2;
                k = k+1;
                iH1(k) = e1; jH1(k) = e2;
                sH1(k) = max(0,rmin-sqrt((i1-i2)^2+(j1-j2)^2));
            end
        end
    end
end
H = sparse(iH1,jH1,sH1);
Hs = sum(H,2);

save(fullfile(out, 'filters.mat'), 'H', 'Hs', 'L', 'nelx', 'nely', '-v7');

%% Gravity load matrix C
I = []; J = []; S = [];
fe = 1/(nely*nelx);
for x = 1:nely
    for y = 1:nelx
        I = [I; (y-1)*(nely+1)+x; (y-1)*(nely+1)+x+1; y*(nely+1)+x; y*(nely+1)+x+1];
        J = [J; (y-1)*nely+x; (y-1)*nely+x; (y-1)*nely+x; (y-1)*nely+x];
        S = [S; fe/4; fe/4; fe/4; fe/4];
    end
end
C = sparse(I, J, S);

save(fullfile(out, 'gravity.mat'), 'C', 'nelx', 'nely', '-v7');

%% Time-field initialization -- all 3 tfield variants (pure, no loop dependence)
ypos = linspace(0, nely, nely);
xpos = linspace(0, nelx, nelx);
[xmesh, ymesh] = meshgrid(xpos, ypos);
pos = [xmesh(:) ymesh(:)];

vec1 = pos - [0, 0];
t1 = sqrt(sum(vec1.*vec1, 2)); t1 = t1 / max(t1);
tfield1 = reshape(t1, nely, nelx);

tfield2 = zeros(nely, nelx);
t2 = linspace(0, 1, nelx);
for i = 1:nelx
    tfield2(:, i) = t2(i);
end

vec3 = pos - [0, nely];
t3 = sqrt(sum(vec3.*vec3, 2)); t3 = t3 / max(t3);
tfield3 = reshape(t3, nely, nelx);

save(fullfile(out, 'timefield.mat'), 'tfield1', 'tfield2', 'tfield3', 'nelx', 'nely', '-v7');

%% Conductivity-estimation neighbor list (N_el, w_el) and WE, as COO triplets
rmin_cond = 3;
N_el = cell(nely*nelx,1);
w_el = cell(nely*nelx,1);
coo_e1 = []; coo_e2 = []; coo_w = [];
for i1 = 1:nelx
    for j1 = 1:nely
        e1 = (i1-1)*nely+j1;
        Nel = []; w = [];
        for i2 = max(i1-(ceil(rmin_cond)-1),1):min(i1+(ceil(rmin_cond)-1),nelx)
            for j2 = max(j1-(ceil(rmin_cond)-1),1):min(j1+(ceil(rmin_cond)-1),nely)
                if rmin_cond-sqrt((i1-i2)^2+(j1-j2)^2) >= 0
                    e2 = (i2-1)*nely+j2;
                    if e2 == e1
                        wv = rmin_cond/rmin_cond;
                    else
                        dist = sqrt((i1-i2)^2+(j1-j2)^2);
                        wv = max(0, rmin_cond-dist)/rmin_cond;
                    end
                    Nel = [Nel e2];
                    w = [w wv];
                    coo_e1 = [coo_e1; e1]; coo_e2 = [coo_e2; e2]; coo_w = [coo_w; wv];
                end
            end
        end
        N_el{e1} = Nel;
        w_el{e1} = w;
    end
end

WE = cell(nely*nelx,1); % kept in scope: the main loop's hotspot section below reads WE{i} directly
coo_we1 = []; coo_we2 = []; coo_we = [];
for i = 1:nely*nelx
    E1 = [N_el{i}];
    We = [];
    for j = 1:length(E1)
        w1 = [w_el{E1(j)}];
        n1 = [N_el{E1(j)}];
        we = w1(find(n1==i));
        We = [We we];
        coo_we1 = [coo_we1; i]; coo_we2 = [coo_we2; E1(j)]; coo_we = [coo_we; we];
    end
    WE{i} = We;
end

save(fullfile(out, 'conductivity_neighbors.mat'), 'coo_e1', 'coo_e2', 'coo_w', ...
     'coo_we1', 'coo_we2', 'coo_we', 'nelx', 'nely', 'rmin_cond', '-v7');

%% Freedom of degree and loads (shared by every FE solve below)
F = sparse(2*(nelx+1)*(nely+1), 1, -1, 2*(nely+1)*(nelx+1), 1);
fixeddofs = 1:2*(nely+1);
alldofs = 1:2*(nely+1)*(nelx+1);
freedofs = setdiff(alldofs, fixeddofs);

%% Standalone FE fixture: solve U for the initial (fixed) density field
x0 = repmat(volfrac, nely, nelx);
xPhys0 = (tanh(beta*eta) + tanh(beta*(x0-eta))) / (tanh(beta*eta) + tanh(beta*(1-eta)));
[c0, dcx0, U0] = Cal_c_ce_whole_with_U(nelx, nely, KE, xPhys0, Emin, Emax, penal, freedofs, F);
save(fullfile(out, 'fem_solve.mat'), 'xPhys0', 'U0', 'c0', 'dcx0', 'nelx', 'nely', '-v7');

%% Main loop -- run the actual optimization for a handful of iterations, saving
%% per-iteration state at every step for compliance/constraints/conductivity/MMA
%% fixtures, plus the full trajectory for the Phase 8 end-to-end test.
x = repmat(volfrac, nely, nelx);
xTilde = x;
xPhys = (tanh(beta*eta) + tanh(beta*(xTilde-eta))) / (tanh(beta*eta) + tanh(beta*(1-eta)));
tPhys = tfield3;
t = tPhys;

xold1 = x(:); xold2 = x(:);
xold1 = [xold1; zeros(nely*nelx, 1)];
xold2 = [xold2; zeros(nely*nelx, 1)];
low = 0; upp = 0; loop = 0; rou = 10;

n = 2*nelx*nely;
m = 1 + 1 + nely + 2*nStage + 1; % vol + continuity + start-points(Nei=1:nely) + 2*stage-vol + hotspot

xPhys_traj = zeros(nely, nelx, nloop+1);
tPhys_traj = zeros(nely, nelx, nloop+1);
xPhys_traj(:,:,1) = xPhys;
tPhys_traj(:,:,1) = tPhys;

c_whole_all = zeros(nloop,1);
dcx_whole_all = zeros(nely,nelx,nloop);
c_grav_all = zeros(nloop,nStage);
dcx_grav_all = zeros(nely*nelx,nStage,nloop);
dct_grav_all = zeros(nely*nelx,nStage,nloop);
dc_all = zeros(nely,nelx,nloop);
dt_all = zeros(nely,nelx,nloop);
fval_all = zeros(m,nloop);
dfdx_all = zeros(m,n,nloop);
K_est_all = zeros(nely*nelx,nloop);
numer_all = zeros(nloop,1);
factor_all = zeros(nloop,1);
tru_max_all = zeros(nloop,1);
df1_all = zeros(nely*nelx,nloop);
dt1_all = zeros(nely*nelx,nloop);
xmma_all = zeros(n,nloop);
low_all = zeros(n,nloop);
upp_all = zeros(n,nloop);
lam_all = zeros(m,nloop);
objf = zeros(nloop,1);
vol = zeros(nloop,1);

beta_max = 128;

while loop < nloop
    loop = loop+1;

    if mod(loop, 30) == 0 && rou < 50
        rou = rou + 5;
    end
    if mod(loop, 50) == 0 && beta <= beta_max
        beta = beta * 2;
    end
    if beta > beta_max
        beta = beta_max;
    end

    dc = zeros(nely, nelx);
    dt = zeros(nely, nelx);

    [c, dcx] = Cal_c_ce_whole(nelx, nely, KE, xPhys, Emin, Emax, penal, freedofs, F);
    obj = c;
    objf(loop) = c;
    c_whole_all(loop) = c;
    dcx_whole_all(:,:,loop) = dcx; % captured now: c/dcx names get reused by the nStage loop below
    dx = beta * (1-tanh(beta*(xTilde-eta)).*tanh(beta*(xTilde-eta))) / (tanh(beta*eta) + tanh(beta*(1-eta)));
    dc(:) = dc(:) + H*(dcx(:).*dx(:)./Hs);

    tP = linspace(0, 1, nStage + 1);
    for i = 1 : nStage
        ti = tP(i+1);
        [c, dcx, dct] = Cal_c_ce_for_gravity(nelx, nely, KE, xPhys, tPhys, ...
            Emin, Emax, penal, ti, C, rou, freedofs);
        c_grav_all(loop,i) = c;
        dcx_grav_all(:,i,loop) = dcx;
        dct_grav_all(:,i,loop) = dct;
        obj = obj + Theta*c;
        dc(:) = dc(:) + Theta*H*(dcx(:).*dx(:)./Hs);
        dt(:) = dt(:) + Theta*H*(dct(:)./Hs);
    end
    dc_all(:,:,loop) = dc;
    dt_all(:,:,loop) = dt;

    df0 = [dc, dt];
    df0dx = df0(:);
    f0val = obj;

    move = 0.01;
    xminx = max(0.0, x(:)-move);
    xmaxx = min(1, x(:)+move);
    tmove = 0.01;
    xmint = max(0.0, t(:)-tmove);
    xmaxt = min(1, t(:)+tmove);
    xmin = [xminx; xmint];
    xmax = [xmaxx; xmaxt];
    xval = [x(:); t(:)];

    fval = sum(sum(xPhys)) / (nelx*nely*volfrac) - 1;
    dv = ones(nely,nelx);
    dv(:) = H*(dv(:).*dx(:)./Hs);
    dfdx = [dv(:)'/(nelx*nely*volfrac), zeros(1, nely*nelx)];
    vol(loop) = sum(sum(xPhys))/(nelx*nely);

    if tfield==1
        Nei = 1;
    else
        Nei = 1:nely;
    end
    LL = L;
    kk = 2*(nely*nelx);
    A = LL*tPhys(:);
    B = A.^2/(nely*nelx);
    fval = [fval; kk*(sum(B)-1.0e-6)];
    dft = kk*2*LL'*A;
    dft = H*(dft./Hs)/(nely*nelx);
    dfdx = [dfdx; zeros(1, nely*nelx), dft'];

    fval = [fval; (tPhys(Nei)' - 1.0e-9)];
    ss = zeros(length(Nei), nely*nelx);
    for ii = 1 : length(Nei)
        ss(ii, Nei(ii)) = 1;
    end
    dfdx = [dfdx; zeros(length(Nei), nely*nelx), (H*(ss'./repmat(Hs, 1, length(Nei))))'];

    percent = 1 / nStage;
    tP = linspace(0, 1, nStage+1);
    for i = 1 : nStage
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

        fval = [fval; -sum(xtJoint(:))/(nelx*nely*volfrac) + i*percent - 1.0e-5];
        dfdx = [dfdx; -dfx(:)', -dft(:)'];
    end

    %% hotspot constraint
    p = 25; q = 3; r = 0.05;
    XPhys = xPhys(:);
    K_est = zeros(nely*nelx,1);
    T_sub = cell(nely*nelx,1);
    N_sub = cell(nely*nelx,1);
    X_sub = cell(nely*nelx,1);
    FT_el = cell(nely*nelx,1);
    DFT_el = cell(nely*nelx,1);
    Nsum2 = zeros(nelx*nely,1);
    rouf = 100;
    TPhys = tPhys(:);
    for l = 1 : nely*nelx
        ti = TPhys(l);
        N_ele = [N_el{l}];
        FT = []; DFT = [];
        for o = 1:length(N_ele)
            FT(o) = (1+exp(rouf*(TPhys(N_ele(o))-ti)))^(-1);
            if TPhys(N_ele(o)) == ti
                DFT(o) = 0;
            else
                DFT(o) = (1+exp(rouf*(TPhys(N_ele(o))-ti)))^(-2)*rouf*exp(rouf*(TPhys(N_ele(o))-ti));
            end
        end
        FT_el{l} = FT;
        DFT_el{l} = DFT;
        Nsum2(l) = sum(FT);
    end

    Nsum3 = zeros(nely*nelx,1);
    for i = 1:nely*nelx
        ti_e = [FT_el{i}];
        w_e = [w_el{i}];
        Nsum3(i) = sum(ti_e.*w_e);
        K_est(i) = sum((XPhys(N_el{i}).^q).*w_el{i}'.*FT_el{i}')/Nsum3(i);
    end
    T_val = 1-K_est;
    cond_p = (T_val.*XPhys.^r).^p;
    sum_cond = sum(cond_p);
    n1 = nelx*nely;
    numer = ((sum_cond/n1)^(1/p));

    if rem(loop,25) == 0
        max_g = max(T_val.*XPhys.^r, [], 'all');
        factor = max_g/numer;
    end
    tru_max = factor*numer;

    fval = [fval; (factor*numer/Tcr)-1];

    N_sub11 = cell(nely*nelx,1);
    N_sub22 = cell(nely*nelx,1);
    for i = 1:nely*nelx
        X_sub{i} = XPhys(N_el{i},1);
        T_sub{i} = T_val(N_el{i},1).*X_sub{i}.^r;
        N_sub{i} = Nsum3(N_el{i},1);
        E1 = [N_el{i}];
        N1 = [N_sub{i}];
        X1 = [X_sub{i}];
        W1 = [WE{i}];
        W2 = [w_el{i}];
        DFT2 = [DFT_el{i}];
        N_sub2 = []; N_sub1 = [];
        for j = 1:length(E1)
            te = [FT_el{E1(j)}];
            de = [DFT_el{E1(j)}];
            n11 = [N_el{E1(j)}];
            F2 = te(find(n11==i));
            DFT1 = de(find(n11==i));
            if E1(j) == i
                N_sub2(j) = -X1(j)^(r)*q*XPhys(i)^(q-1)*F2*W1(j)/N1(j)+(1-K_est(i))*r*X1(j)^(r-1);
                N_sub1(j) = -X1(j)^r*(sum(X1.^q.*W2'.*DFT2')/N1(j)-K_est(i)*sum(W2.*DFT2)/N1(j));
            else
                N_sub2(j) = -X1(j)^(r)*q*XPhys(i)^(q-1)*F2*W1(j)/N1(j);
                N_sub1(j) = -(-XPhys(i)^q*W1(j)/N1(j)+K_est(E1(j))*W1(j)/N1(j))*X1(j)^r*DFT1;
            end
        end
        N_sub22{i} = N_sub2;
        N_sub11{i} = N_sub1;
    end

    df1 = zeros(1, nely*nelx);
    dt1 = zeros(1, nely*nelx);
    for i = 1:nely*nelx
        numer1 = ((sum_cond/n1)^((1/p)-1));
        denom1 = n1*Tcr;
        cond_arr1 = sum((T_sub{i}.^(p-1)).*(N_sub11{i})');
        cond_arr2 = sum((T_sub{i}.^(p-1)).*(N_sub22{i})');
        df1(i) = (factor*numer1/denom1)*cond_arr2;
        dt1(i) = (factor*numer1/denom1)*cond_arr1;
    end

    df1(:) = H*(df1(:).*dx(:)./Hs);
    dt1(:) = H*(dt1(:)./Hs);
    dfdx = [dfdx; df1, dt1];

    K_est_all(:,loop) = K_est;
    numer_all(loop) = numer;
    factor_all(loop) = factor;
    tru_max_all(loop) = tru_max;
    df1_all(:,loop) = df1;
    dt1_all(:,loop) = dt1;
    fval_all(:,loop) = fval;
    dfdx_all(:,:,loop) = dfdx;

    if loop == 1
        xval_1 = xval; xmin_1 = xmin; xmax_1 = xmax;
        xold1_1 = xold1; xold2_1 = xold2;
        f0val_1 = f0val; df0dx_1 = df0dx;
        fval_1 = fval; dfdx_1 = dfdx;
        low_1 = low; upp_1 = upp;
    end

    %% Optimizing with MMA solver
    mdof = 1:m;
    a0 = 1;
    a = zeros(m,1);
    c_ = ones(m,1)*2500;
    d = zeros(m,1);

    [xmma, ymma, zmma, lam, xsi, xsi_e, mu, zet, s, low, upp] = ...
        mmasub(m, n, loop, xval, xmin, xmax, xold1, xold2, ...
        f0val, df0dx, fval(mdof), dfdx(mdof,:), low, upp, a0, a, c_, d);

    xmma_all(:,loop) = xmma;
    low_all(:,loop) = low;
    upp_all(:,loop) = upp;
    lam_all(:,loop) = lam;

    xnew = reshape(xmma, nely, []);
    xold2 = xold1;
    xold1 = xval;
    s = xnew(:, 1:nelx);

    xTilde(:) = (H*s(:))./Hs;
    xPhys = (tanh(beta*eta) + tanh(beta*(xTilde-eta))) / (tanh(beta*eta) + tanh(beta*(1-eta)));
    x = s;

    t = xnew(:, nelx+1:end);
    tPhys = t;
    tPhys(:) = (H*t(:))./Hs;

    xPhys_traj(:,:,loop+1) = xPhys;
    tPhys_traj(:,:,loop+1) = tPhys;
end

save(fullfile(out, 'compliance.mat'), 'c_whole_all', 'dcx_whole_all', ...
     'c_grav_all', 'dcx_grav_all', 'dct_grav_all', 'nelx', 'nely', 'nStage', '-v7');

save(fullfile(out, 'constraints.mat'), 'fval_all', 'dfdx_all', 'm', 'n', ...
     'nelx', 'nely', 'nStage', 'volfrac', 'tfield', '-v7');

save(fullfile(out, 'conductivity.mat'), 'K_est_all', 'numer_all', 'factor_all', ...
     'tru_max_all', 'df1_all', 'dt1_all', 'nelx', 'nely', '-v7');

save(fullfile(out, 'mma.mat'), 'xval_1', 'xmin_1', 'xmax_1', 'xold1_1', 'xold2_1', ...
     'f0val_1', 'df0dx_1', 'fval_1', 'dfdx_1', 'low_1', 'upp_1', 'm', 'n', ...
     'xmma_all', 'low_all', 'upp_all', 'lam_all', 'nelx', 'nely', '-v7');

save(fullfile(out, 'e2e.mat'), 'xPhys_traj', 'tPhys_traj', 'objf', 'vol', ...
     'tru_max_all', 'nelx', 'nely', 'nStage', 'volfrac', 'Theta', 'Tcr', ...
     'tfield', 'nloop', '-v7');

disp('Fixture generation complete.');

%% Subfunctions -- verbatim copies of the local functions from
%% conductivity_estimation_stto_main.m (lines 605-665 as of this writing), plus
%% one thin wrapper that also returns U for the standalone FE fixture.

function [c,dcx] = Cal_c_ce_whole(nelx, nely, KE, xPhys, Emin, Emax, penal, freedofs, F)

nodenrs = reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx);
edofVec = reshape(2*nodenrs(1:end-1,1:end-1)+1,nelx*nely,1);
edofMat = repmat(edofVec,1,8)+repmat([0 1 2*nely+[2 3 0 1] -2 -1],nelx*nely,1);
iK = reshape(kron(edofMat,ones(8,1))',64*nelx*nely,1);
jK = reshape(kron(edofMat,ones(1,8))',64*nelx*nely,1);
sK = reshape(KE(:)*(Emin+xPhys(:)'.^penal*(Emax-Emin)),64*nelx*nely,1);
K = sparse(iK,jK,sK);
K = (K + K') / 2;

U = zeros(2*(nely+1)*(nelx+1), 1);
U(freedofs) = K(freedofs, freedofs)\F(freedofs);
ce = zeros(nely, nelx);
ce(1 : nely, 1 : nelx) = reshape(sum((U(edofMat)*KE).*U(edofMat),2), nely, nelx);
c = sum(sum((Emin+xPhys.^penal*(Emax-Emin)).*ce));
dcx = -penal*(Emax-Emin)*xPhys.^(penal-1).*ce;
end

function [c,dcx,U] = Cal_c_ce_whole_with_U(nelx, nely, KE, xPhys, Emin, Emax, penal, freedofs, F)
[c, dcx] = Cal_c_ce_whole(nelx, nely, KE, xPhys, Emin, Emax, penal, freedofs, F);
nodenrs = reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx);
edofVec = reshape(2*nodenrs(1:end-1,1:end-1)+1,nelx*nely,1);
edofMat = repmat(edofVec,1,8)+repmat([0 1 2*nely+[2 3 0 1] -2 -1],nelx*nely,1);
iK = reshape(kron(edofMat,ones(8,1))',64*nelx*nely,1);
jK = reshape(kron(edofMat,ones(1,8))',64*nelx*nely,1);
sK = reshape(KE(:)*(Emin+xPhys(:)'.^penal*(Emax-Emin)),64*nelx*nely,1);
K = sparse(iK,jK,sK);
K = (K + K') / 2;
U = zeros(2*(nely+1)*(nelx+1), 1);
U(freedofs) = K(freedofs, freedofs)\F(freedofs);
end

function [c, dcx, dct] = Cal_c_ce_for_gravity(nelx, nely, KE ...
    , xPhys, tPhys, ...
    Emin, Emax, penal, ti, C, lamda, freedofs)

ft = 1 - (tanh(lamda*ti) + tanh(lamda*(tPhys - ti)))/(tanh(lamda*ti) + tanh(lamda*(1-ti)));
dfdt = -(lamda*(tanh(lamda*(tPhys - ti)).^2 - 1))/(tanh(lamda*(ti - 1)) - tanh(lamda*ti));
xtJoint = xPhys.*ft;

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

U = zeros(2*(nely+1)*(nelx+1), 1);
U(freedofs) = K(freedofs, freedofs)\F(freedofs);

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
