function [Kappa,Nsum]=conductivity_est_function_stt(nelx,nely,xPhys,N_el,TPhys3,rouf,w_el)
q=3;


for l = 1 : nely*nelx
        ti = TPhys3(l);
        N_ele=[N_el{l}];
        for o=1:length(N_ele)
%         FT(o) = 1 - (tanh(rouf*ti) + tanh(rouf*(TPhys3(N_ele(o)) - ti)))/(tanh(rouf*ti) + tanh(rouf*(1-ti)));
         FT(o)=(1+exp(rouf*(TPhys3(N_ele(o))-ti)))^(-1);
        end
        FT_el3{l}=FT;
        Nsum(l)=sum(FT);
        FT=[];
end
XPhys=xPhys(:);

for i=1:nely*nelx
   Kappa(i)=sum((XPhys(N_el{i}).^q).*w_el{i}'.*FT_el3{i}')/sum(FT_el3{i}.*w_el{i});
end

% for i=1:nely*nelx
%    Kappa(i)=sum((XPhys(N_el{i}).^q).*FT_el3{i}')/sum(FT_el3{i});
% end