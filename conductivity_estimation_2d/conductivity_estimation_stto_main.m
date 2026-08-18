 % Function: Controlling local overheating in space-time optimization
% Author:  Manabendra Das (m.n.s.das@tudelft.nl)
% Version: 2023-08-25
% volfrac: volume constraint
% nStage:  the number of layers
% nloop:   the number of iterations
% nely:    dimension in y-axis
% nelx:    dimension in x-axis
% Theta:   weight factor alpha in objective function

% [xPhys, tPhys, data] = Space_Time_TopOpt_Gravity(120, 40, 600, 8, 0.6, 0.1);
%function [xPhys, tPhys, data] = Space_Time_TopOpt_Gravity(nelx, nely, nloop, nStage, volfrac, Theta)
% close all;
nelx=180;
nely=60;
nloop=800;
nStage=8;
volfrac=0.5;
Theta=0.1;
Tcr=0.8;
factor=1;
tfield=3; % 1: top left corner, 2: left edge, 3: bottom left corner
beta_max=128;

tic
fopen('screen.txt','w');
diary screen.txt;
warning('off')
% Connectivity matrix for continuity 
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

% Material properties
Emax = 1;
Emin = 1e-9;
nu = 0.3;

% Initialization several parameters
penal = 3;      % stiffness penalty 
rmin = 4;     % density filter radius

% Stiffness matrix for element
A11 = [12  3 -6 -3;  3 12  3  0; -6  3 12 -3; -3  0 -3 12];
A12 = [-6 -3  0  3; -3 -6 -3 -6;  0 -3 -6  3;  3 -6  3 -6];
B11 = [-4  3 -2  9;  3 -4 -9  4; -2 -9 -4 -3;  9  4 -3 -4];
B12 = [ 2 -3  4 -9; -3  2  9 -2;  4  9  2  3; -9 -2  3  2];
KE = 1/(1-nu^2)/24*([A11 A12;A12' A11]+nu*[B11 B12;B12' B11]);

% Density filter
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

% Definition of gravity force
I = []; J = []; S = [];
fe = 1 / (nely*nelx);
for x = 1 : nely
    for y = 1 : nelx
        I = [I; (y-1)*(nely+1) + x; (y-1)*(nely+1)+x+1; y*(nely+1)+x; y*(nely+1)+x+1];
        J = [J; (y-1)*nely + x; (y-1)*nely + x; (y-1)*nely + x; (y-1)*nely + x];
        S = [S; fe/4; fe/4; fe/4; fe/4];
    end
end
C = sparse(I, J, S);

%
beta = 1;       % projection parameter
eta = 0.5;      % projection threshold, fixed at 0.5

% Initialize density field
x = repmat(volfrac,nely,nelx);
% x=rand(nely,nelx);
xTilde = x;
xPhys = (tanh(beta*eta) + tanh(beta*(xTilde-eta))) / (tanh(beta*eta) + tanh(beta*(1-eta)));

% Initialize time field (start from left boundary)
if tfield==1
ypos = linspace(0, nely, nely);
xpos = linspace(0, nelx , nelx);
[xmesh, ymesh] = meshgrid(xpos, ypos);
pos = [xmesh(:) ymesh(:)];
start_pos = [0, 0];
vec = pos - start_pos;
dis2 = sum(vec.*vec, 2);
t = sqrt(dis2) / max(sqrt(dis2));
tPhys = reshape(t,nely, nelx);
elseif tfield==2
    tPhys = zeros(nely, nelx);
t = linspace(0, 1, nelx);
for i = 1 : nelx
    tPhys(:, i) = t(i);
end
else
    ypos = linspace(0, nely, nely);
xpos = linspace(0, nelx , nelx);
[xmesh, ymesh] = meshgrid(xpos, ypos);
pos = [xmesh(:) ymesh(:)];
start_pos = [0, nely];
vec = pos - start_pos;
dis2 = sum(vec.*vec, 2);
t = sqrt(dis2) / max(sqrt(dis2));
tPhys = reshape(t,nely, nelx);
end

% Freedom of degree and loads
F = sparse( 2 * (nelx+1)*(nely+1), 1, -1, 2*(nely+1)*(nelx+1), 1);
fixeddofs = 1 : 2*(nely+1);
alldofs = 1 : 2*(nely+1)*(nelx+1);
freedofs = setdiff(alldofs, fixeddofs);

% Initialization
t = tPhys;
xold1 = x(:);
xold2 = x(:);
xold1 = [xold1; zeros(nely*nelx, 1)];
xold2 = [xold2; zeros(nely*nelx, 1)];
low = 0;
upp = 0;
loop = 0;
rou = 10;
% edit 20201214 for iteration plot
objf = zeros(1000,1);
% consf = zeros(1000,1);
objint = zeros(1000,nStage);
vol = zeros(1000,1);

% conductivity estimation filter
rmin=12;
iH1 = ones(nelx*nely*((ceil(rmin)-1)+1)^2,1);
jH1 = ones(size(iH1));
sH1 = zeros(size(iH1));
k = 0;
Kappa=zeros(nely*nelx,1);
dsum=0;
Nel=[];
Eel=[];
Nsum1=0;
Nsum=zeros(nelx*nely,1);
K_sub=[];
N_sub=[];
N_el=cell(nely*nelx,1);
w_el=cell(nely*nelx,1);
W=0;
lambda1=1;
lambda=0.1;
per=1;
for i1 = 1:nelx
    for j1 = 1:nely
        e1 = (i1-1)*nely+j1;
        for i2 = max(i1-(ceil(rmin)-1),1):min(i1+(ceil(rmin)-1),nelx)
            for j2 = max(j1-(ceil(rmin)-1),1):min(j1+(ceil(rmin)-1),nely)
               if rmin-sqrt((i1-i2)^2+(j1-j2)^2)>=0
                e2 = (i2-1)*nely+j2;
                 if e2==e1 
                     k = k+1;
                    sH1(k)=rmin;
                     w(k)=sH1(k)/rmin;
                else   
                k = k+1;
                dist=sqrt((i1-i2)^2+(j1-j2)^2);
                  sH1(k)= max(0,rmin-dist);
                w(k)=sH1(k)/rmin; 
                 end
                    Nel=[Nel e2];
                    Nsum1=sum(w);
                else
                    continue
                end
            end
        end
         Nsum(e1)=Nsum1;
         N_el{e1}=Nel;
         w_el{e1}=w;
         Nsum1=0;
         Nel=[];
         w=0;
         k=0;
    end
end

WE=cell(nely*nelx,1);
We=[];
for i=1:nely*nelx
     E1=[N_el{i}];
     for j=1:length(E1)
   w1=[w_el{E1(j)}];
   n1=[N_el{E1(j)}];
   we=w1(find(n1==i));
   We=[We we];
     end
     WE{i}=We;
     We=[];
end
%%
while loop < nloop
    loop = loop+1;
    
    % Parameter for projection on time field
    if  mod(loop, 30) == 0 && rou < 50
        rou = rou + 5;
    end
    
    % Parameter for projection on density field
    if mod(loop, 50) == 0 && beta <= beta_max
        beta = beta * 2;
    end
   if(beta>beta_max)
       beta = beta_max;
   end
    
    % Ojectives and sensitivities
    dc = zeros(nely, nelx);
    dt = zeros(nely, nelx);
    
    % Compliance of the whole structure
    [c, dcx] = Cal_c_ce_whole(nelx, nely, KE, xPhys, Emin, Emax, penal, freedofs, F);
    obj = c;
    objf(loop) = c;
    dx = beta * (1-tanh(beta*(xTilde-eta)).*tanh(beta*(xTilde-eta))) / (tanh(beta*eta) + tanh(beta*(1-eta)));
    dc(:) = dc(:) + H*(dcx(:).*dx(:)./Hs);
    
    % Compliances of the intermediate structures
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
    
    % Lower and upper bounds for timefield and densityfield
    move = 0.01;    
    xminx=max(0.0, x(:)-move);
    xmaxx=min(1, x(:)+move);
    tmove = 0.01;
    xmint=max(0.0, t(:)-tmove);
    xmaxt=min(1, t(:)+tmove);
    xmin = [xminx; xmint];
    xmax = [xmaxx; xmaxt];
    xval = [x(:); t(:)];
    
    % Global volume constraint
    fval = sum(sum(xPhys)) / (nelx*nely*volfrac) - 1;
    print_out = fval(end);
    dv = ones(nely,nelx);
    dv(:) = H*(dv(:).*dx(:)./Hs);
    dfdx = [dv(:)'/(nelx*nely*volfrac), zeros(1, nely*nelx)];
    vol(loop) = sum(sum(xPhys))/(nelx*nely);
    
    % Continuity constraint
    if tfield==1
    Nei = 1;
    else
        Nei=1:nely;
    end
    LL = L;
    kk = 2*(nely*nelx); % controlling the smoothness of the time field
    A = LL*tPhys(:);
    B = A.^2/(nely*nelx);
    fval = [fval; kk*(sum(B)-1.0e-6)];
    print_out = [print_out; fval(end)/kk];
    dft = kk*2*LL'*A;
    dft = H*(dft./Hs)/(nely*nelx);
    dfdx = [dfdx; zeros(1, nely*nelx), dft'];
    
    % Start points
    fval = [fval; (tPhys(Nei)' - 1.0e-9)];
    ss = zeros(length(Nei), nely*nelx);
    for ii = 1 : length(Nei)
        ss(ii, Nei(ii)) = 1;
    end
    dfdx = [dfdx; zeros(length(Nei), nely*nelx), (H*(ss'./repmat(Hs, 1, length(Nei))))'];
    
    % Volume constraints for intermediate structures
    percent = 1 / nStage;
    tP = linspace(0, 1, nStage+1);
    for i = 1 : nStage
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
    %% hotspot constraint
p=25;
q=3;
r=0.05; 

XPhys=xPhys(:);
K_est=zeros(nely*nelx,1);
T_sub=cell(nely*nelx,1);
N_sub=cell(nely*nelx,1);
X_sub=cell(nely*nelx,1);
FT_el=cell(nely*nelx,1);
 DFT_el=cell(nely*nelx,1);
FT1=zeros(nelx*nely,1);
Nsum2=zeros(nelx*nely,1);
rouf=100;
TPhys=tPhys(:);
 for l = 1 : nely*nelx
        ti = TPhys(l);
        N_ele=[N_el{l}];
        for o=1:length(N_ele)
        FT(o)=(1+exp(rouf*(TPhys(N_ele(o))-ti)))^(-1);
        if TPhys(N_ele(o))==ti
            DFT(o)=0;
        else
            DFT(o)= (1+exp(rouf*(TPhys(N_ele(o))-ti)))^(-2)*rouf*exp(rouf*(TPhys(N_ele(o))-ti));
        end
        end
        FT_el{l}=FT;
         DFT_el{l}=DFT;
        Nsum2(l)=sum(FT);
        FT=[];
        DFT=[];
 end

% xdum=zeros(nely,nelx);
% xdum(N_el{l})=FT_el{l}.*w_el{l};
%  figure(1);
%         colormap(gray); imagesc(-xdum, [-1 0]); axis equal; axis tight; axis off; drawnow;


Nsum3=zeros(nely*nelx,1);
for i=1:nely*nelx
    ti_e=[FT_el{i}];
    w_e=[w_el{i}];
    Nsum3(i)=sum(ti_e.*w_e);
   K_est(i)=sum((XPhys(N_el{i}).^q).*w_el{i}'.*FT_el{i}')/Nsum3(i);
end
T_val=1-K_est;
cond_p=(T_val.*XPhys.^r).^p;
sum_cond=sum(cond_p);
n1=nelx*nely;
numer=((sum_cond/n1)^(1/p));
% Numer(loop)=numer;

% APPLYING P-NORM SCALING FACTOR
  if rem(loop,25)==0
     max_g=max(T_val.*XPhys.^r,[],'all');
     factor=max_g/numer;
  end
  tru_max=factor*numer;
  Tru_m(loop)=tru_max;

fval=[fval;(factor*numer/Tcr)-1];
 fval1=(numer/Tcr)-1;
 
N_sub11=cell(nely*nelx,1);
N_sub22=cell(nely*nelx,1);
 for i=1:nely*nelx
    X_sub{i}=XPhys(N_el{i},1);
    T_sub{i}=T_val(N_el{i},1).*X_sub{i}.^r;
    N_sub{i}=Nsum3(N_el{i},1);
    E1=[N_el{i}];
    F1=[FT_el{i}];
     N1=[N_sub{i}];
     X1=[X_sub{i}];
     W1=[WE{i}];
     W2=[w_el{i}];
     DFT2=[DFT_el{i}];
    for j=1:length(E1)
          te=[FT_el{E1(j)}];
           de=[DFT_el{E1(j)}];
   n11=[N_el{E1(j)}];
   F2=te(find(n11==i));
     DFT1=de(find(n11==i));
        if E1(j)==i      
           N_sub2(j)=-X1(j)^(r)*q*XPhys(i)^(q-1)*F2*W1(j)/N1(j)+(1-K_est(i))*r*X1(j)^(r-1);
           N_sub1(j)=-X1(j)^r*(sum(X1.^q.*W2'.*DFT2')/N1(j)-K_est(i)*sum(W2.*DFT2)/N1(j));
        else
           N_sub2(j)=-X1(j)^(r)*q*XPhys(i)^(q-1)*F2*W1(j)/N1(j);
           N_sub1(j)=-(-XPhys(i)^q*W1(j)/N1(j)+K_est(E1(j))*W1(j)/N1(j))*X1(j)^r*DFT1;
        end
      
    end
    N_sub22{i}=N_sub2;
    N_sub11{i}=N_sub1;
     N_sub2=[];
     N_sub1=[];
 end

for i=1:nely*nelx
numer1=((sum_cond/n1)^((1/p)-1));
denom1=n1*Tcr;
cond_arr1(i)=sum((T_sub{i}.^(p-1)).*(N_sub11{i})');
cond_arr2(i)=sum((T_sub{i}.^(p-1)).*(N_sub22{i})');
 df1(i)=(factor*numer1/denom1)*cond_arr2(i);
 dt1(i)=(factor*numer1/denom1)*cond_arr1(i);
end

df1(:)=H*(df1(:).*dx(:)./Hs);
dt1(:)=H*(dt1(:)./Hs);
dfdx = [dfdx; df1, dt1];

  
 %% sensitivity FD check
% h=1e-9;
% xPhys3=xPhys;
% dt11=zeros(nely,nelx);
% for i=13
%       for j=[1,12,7,23]
% %      for j=1:nely
%  xPhys3(j,i)=xPhys(j,i)+h;
%  [K_est1,Nsum]=conductivity_est_function_st(nelx,nely,xPhys3,N_el,TPhys,rouf,w_el,Nsum3);
%  cond_p1=((1-K_est1).*(xPhys3(:)'.^r)).^p;
% sum_cond1=sum(cond_p1);
% n1=nelx*nely;
% numer_=(sum_cond1/n1)^(1/p);
% fval_k=(numer_/Tcr)-1;
% dt11(j,i)=(fval_k-fval1)/h;
% xPhys3=xPhys;
%     end 
% end
%    DT11(loop,:)=[dt11(361),dt11(372),dt11(367),dt11(383)];
% % % 
% h=1e-9;
% TPhys3=tPhys;
% dt22=zeros(nely,nelx);
% for i=13
%      for j=[1,12,7,23]
% %      for j=1:nely
%  TPhys3(j,i)=tPhys(j,i)+h;
%  [K_est2,Nsum]=conductivity_est_function_stt(nelx,nely,xPhys,N_el,TPhys3,rouf,w_el);
%  cond_p2=((1-K_est2).*xPhys(:)'.^r).^p;
% sum_cond2=sum(cond_p2);
% n1=nelx*nely;
% numer_2=(sum_cond2/n1)^(1/p);
% fval_k2=(numer_2/Tcr)-1;
% dt22(j,i)=(fval_k2-fval1)/h;
% TPhys3=tPhys;
%     end
% end
%  DT22(loop,:)=[dt22(361),dt22(372),dt22(367),dt22(383)];
    %% Optimizing with MMA solver
    m=length(fval);
    mdof = 1:m;
    a0 = 1;
    a = zeros(m,1);
    c_ = ones(m,1)*2500;
    d = zeros(m,1);
    
    [xmma, ymma, zmma, lam, xsi, eta_, mu, zet, s, low, upp] = ...
        mmasub(m, n, loop, xval, xmin, xmax, xold1, xold2,...
        f0val, df0dx, fval(mdof), dfdx(mdof,:),low, upp, a0, a, c_, d);
   
    xnew = reshape(xmma, nely, []);
    xold2 = xold1;
    xold1 = xval;
    s = xnew(:, 1:nelx);
    
 
     change = 1;

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
        ' Vol.: ' sprintf('%6.3f',sum(sum(xPhys))/(nelx*nely)) ...
        ' Tm.: ' sprintf('%7.3f',tru_max)]);
    
    if mod(loop, 10) == 0
        figure(1);
        colormap(gray); imagesc(-xPhys, [-1 0]); axis equal; axis tight; axis off; drawnow;
        hold on
        
        figure(2);
        colormap(parula);
        imagesc(tPhys); axis equal; axis tight; axis off; drawnow;
        % % imagesc: The row and column indices of the elements determine the centers of the corresponding pixels
    end
    
    % edit 20201214 to plot the convergence of obj and cons
    % The convergence plot of compliance
%     figure(3)
%     myfontsize = 24;
%     labelsize = 26;
%     plot(1:data.loop,data.objf(1:data.loop));
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
% save('sttot1')
%% edit 20201213 add the combination figure of layout and time field
draw_boundary(tPhys,nStage);
draw_combination(xPhys,tPhys,nStage,1.0e-1);
toc
diary off
%%end%%
 B=reshape(K_est,nely,nelx);
% T=(1-B).*xPhys;
% draw_combination1(xPhys,T,1.0e-1);

for i=1:length(XPhys(:))
    if XPhys(i)>0.5
        XPhys(i)=1;
    else
        XPhys(i)=0;
    end
end
XPhys=reshape(XPhys,nely,nelx);
T1=(1-B).*XPhys;
draw_combination1(XPhys,T1,1.0e-1);
% % draw_combination3(xPhys,tPhys,nStage,1.0e-1);
% %  
%  loopn=1:nloop;
% % %     figure
% % %     plot(loopn,T_max,loopn,Numer,loopn,Tru_m);
% % 


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
function [c, dcx, dct] = Cal_c_ce_for_gravity(nelx, nely, KE ...
    , xPhys, tPhys, ...
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


