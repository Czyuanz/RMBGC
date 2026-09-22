function outputFile = prepare_zscore_dataset(inputFile,outputFile)
%PREPARE_ZSCORE_DATASET Standardize each view by feature and save Xs/Y.
if nargin<2||strlength(string(outputFile))==0
    [folder,name]=fileparts(inputFile);
    outputFile=fullfile(folder,[name '_zscore.mat']);
end
assert(string(inputFile)~=string(outputFile), ...
    'prepare_zscore_dataset:SameFile','Input and output files must differ.');

S=load(inputFile);
if isfield(S,'Xs'),Xs=S.Xs;elseif isfield(S,'X'),Xs=S.X;else,error('prepare_zscore_dataset:Features','MAT file must contain Xs or X.');end
if isfield(S,'Y'),Y=S.Y;elseif isfield(S,'label'),Y=S.label;else,error('prepare_zscore_dataset:Labels','MAT file must contain Y or label.');end
if ~iscell(Xs),Xs={Xs};end
Xs=Xs(:);Y=Y(:);n=numel(Y);

for v=1:numel(Xs)
    Xv=double(Xs{v});
    if size(Xv,1)~=n&&size(Xv,2)==n,Xv=Xv';end
    assert(size(Xv,1)==n,'prepare_zscore_dataset:ViewRows','View %d does not contain one row per sample.',v);
    assert(all(isfinite(nonzeros(Xv))),'prepare_zscore_dataset:NonFinite','View %d contains NaN or Inf.',v);
    mu=mean(Xv,1);sigma=std(Xv,0,1);
    sigma(sigma<1e-12)=1; % Constant features become zero after centering.
    Xs{v}=(Xv-mu)./sigma;
end

[~,~,Y]=unique(Y,'sorted');
save(outputFile,'Xs','Y','-v7.3');
end
