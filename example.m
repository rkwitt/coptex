% Make MATLAB code available
setuppath;
% Path to directory where all matlab files reside
basedir = '/Users/rkwitt/Remote/coptex-read-only';
% Directory where to dump the features
dumpdir = '/tmp/x';


% Load sample textures (from VisTex dataset)
data = texload('Testset','basedir', ...
    fullfile(basedir,'testimages'),...
    'debug',true,'from',1,'to',2);

models = ctbrun(data);

[succ] = mkdir(dumpdir);

% Dump models to hdd (dumpdir)
dump_copuladata(models,dumpdir,0);

% Next, run the copll binary, then run the 
% uncommented code below to get a retrieval acc.

%fid = fopen(fullfile(dumpdir,'dist.bin'));
%A = fread(fid,'double');
%A = reshape(A,[32 32])';

%rr = generic_rrate(A,16,'descend');
%acc = evalir(rr);
%disp(acc);
