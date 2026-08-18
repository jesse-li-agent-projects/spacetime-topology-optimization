function [Kappa,Nsum]=conductivity_est_function_st(nelx,nely,xPhys3,N_el,TPhys,rouf,w_el,Nsum3)
q=3;


for l = 1 : nely*nelx
        ti = TPhys(l);
        N_ele=[N_el{l}];
        for o=1:length(N_ele)
%         FT(o) = 1 - (tanh(rouf*ti) + tanh(rouf*(TPhys(N_ele(o)) - ti)))/(tanh(rouf*ti) + tanh(rouf*(1-ti)));
          FT(o)=(1+exp(rouf*(TPhys(N_ele(o))-ti)))^(-1);
        end
        FT_el{l}=FT;
        Nsum(l)=sum(FT);
        FT=[];
end

XPhys3=xPhys3(:);
for i=1:nely*nelx
    ti_e=[FT_el{i}];
    w_e=[w_el{i}];
    Nsum3(i)=sum(ti_e.*w_e);
   Kappa(i)=sum((XPhys3(N_el{i}).^q).*w_el{i}'.*FT_el{i}')/Nsum3(i);
end

% for i=1:nely*nelx
%    Kappa(i)=sum((XPhys3(N_el{i}).^q).*FT_el{i}')/sum(FT_el{i});
% end
