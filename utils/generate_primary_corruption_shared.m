function Bout=generate_primary_corruption_shared(B,Y,tau,seed)
%GENERATE_PRIMARY_CORRUPTION_SHARED Apply equal-class anchor-mass corruption.
[~,~,g]=unique(Y(:),'sorted');counts=accumarray(g,1);
b=max(1,round(tau*min(counts)));V=numel(B);Bout=B;classes=unique(Y(:),'sorted');
assert(all(counts>=b));
for k=1:numel(classes)
 rng(safe_seed(8300003+10007*seed+1009*k),'twister');
 idx=find(g==k);order=idx(randperm(numel(idx)));chosen=order(1:b);
 for jj=1:b
  i=chosen(jj);rng(safe_seed(9100009+10007*seed+1009*k+37*jj),'twister');
  views=randperm(V,max(1,round(V/2)));
  for v=views
   old=Bout{v}(i,:);[~,nz,w]=find(old);degree=numel(nz);m=size(old,2);
   h=min([degree-1,m-degree,max(1,round(.125*degree))]);
   assert(h>=1,'PrimaryCorruption:NoAdmissibleRewire','Row support does not admit a safe rewire.');
   take=randperm(degree,h);available=setdiff(1:size(old,2),nz);dst=available(randperm(numel(available),h));
   keep=true(degree,1);keep(take)=false;keepW=w(keep);keepW=.1*keepW/sum(keepW);
   Bout{v}(i,:)=sparse(1,[nz(keep)';dst(:)'],[keepW(:);.9*ones(h,1)/h],1,size(old,2));
  end
 end
end
end
function s=safe_seed(x),s=mod(round(double(x)),2^32-1);if s<=0,s=s+1;end,end
