function fv = dtcwtex(X)
    fv = [];
    for i=1:length(X)
        fv = [fv; X{i}.features];
    end
end
