function data = texload(dbname,varargin)
    p = inputParser;
    p.addRequired('dbname',@isstr);
    p.addParamValue('basedir','/tmp/',@isstr);
    p.addParamValue('nsubs',16,@(x)x>=1);
    p.addParamValue('parent',false,@islogical);
    p.addParamValue('from',1,@(x)x>=1);
    p.addParamValue('to',1,@(x)x>=1);
    p.addParamValue('ending','tif',@isstr);
    p.addParamValue('rotate',0,@(x)x>=0 && x<= 359);
    p.addParamValue('rrotate',[],@(x)~isempty(x));
    p.addParamValue('pixel', 128, @(x)x>=64);
    p.addParamValue('debug',false,@islogical);
    p.parse(dbname,varargin{:});
    
    dirname = p.Results.basedir;    % directory where images reside
    nsubs = p.Results.nsubs;        % number of subimages per parent
    parent = p.Results.parent;      % process only parent images (e.g. build queries)
    texfrom = p.Results.from;       % process parent from textfrom to ...
    texto = p.Results.to;           % texto ;)
    rotate = p.Results.rotate;      % rotate subimage from parent by 'rotate' degrees
    rrotate = p.Results.rrotate;    % random rotation in [0,359]
    pixel = p.Results.pixel;        % box to crop from parent for processing (e.g. rotating)
    debug = p.Results.debug;        % print debug info
    ending = p.Results.ending;      % image file ending
    progname = 'texload';
    
    data = {};
    name = eval(dbname);
    ntexs = length(name);
    if (parent)
        nsubs = 1;
    end
    cnt = 1;
    if (debug)
        fprintf('[%s]: loading %d textures\n', progname, (texto-texfrom+1)*nsubs);
    end
    for p = texfrom:texto
        if p > ntexs
            break
        end
        if (parent)
            file = sprintf('%s/%s.%s', dirname, name{p},ending);
            im  = imread(file);
            phi = rotate;
            if (phi>0)
                phi = rotate;
                im = imrotate(im,phi);
            elseif (length(rrotate) == 2)
                assert(rrotate(1) < rrotate(2));
                phi = randsample([rrotate(1):rrotate(2)],1);
                im = imrotate(im,phi);
            end
            sz = size(im);
            height = sz(1);
            width = sz(2);
            im = im(round(height/2)-pixel/2:round(height/2)+pixel/2-1, ...
                round(width/2)-pixel/2:round(width/2)+pixel/2-1, :);
            data{cnt}.image = im;
            data{cnt}.idx = cnt*16;
            data{cnt}.filename = file;
            data{cnt}.rotate = phi;
            data{cnt}.dim = size(im,3);
            cnt = cnt + 1;
        else
            for q=1:nsubs
                file = sprintf('%s/%s.%02d.%s', dirname, name{p}, q,ending);
                data{cnt}.image = imread(file);
                data{cnt}.dim = size(data{cnt}.image,3);
                data{cnt}.idx = cnt;
                data{cnt}.filename = sprintf('%s.%02d',name{p},q);
                cnt = cnt + 1;
            end
        end
    end
    if (debug)
        fprintf('[%s]: read %d images\n',progname, cnt-1);
    end
end
