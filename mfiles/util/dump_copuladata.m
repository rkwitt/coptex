function dump_copuladata(data,where,offset) 
    if offset == 0
      flist = fopen(fullfile(where,'filelist.txt'),'wt');
    else
      flist = fopen(fullfile(where,'filelist.txt'),'at');
    end
    for i=1:length(data)
        fprintf(flist,'model%d\n',i+offset-1);
        
        datastr = sprintf('model%d.data',i+offset-1);
        fiddata = fopen(fullfile(where,datastr),'wb');
        fwrite(fiddata,data{i}.X,'double');
        fclose(fiddata);
        
        Rhostr = sprintf('model%d.Rho',i+offset-1);
        Rhofid = fopen(fullfile(where,Rhostr),'wb');
        fwrite(Rhofid,data{i}.Rho,'double');
        fclose(Rhofid);
        
        marginstr = sprintf('model%d.margins',i+offset-1);
        marginfid = fopen(fullfile(where,marginstr),'wb');
        fwrite(marginfid,data{i}.margins,'double');
        fclose(marginfid);
        
        if (data{i}.nu ~= 0) 
            nustr = sprintf('model%d.nu',i+offset-1);
            nufid = fopen(fullfile(where,nustr),'wb');
            fwrite(nufid,data{i}.nu,'double');
            fclose(nufid);
        end
    end
    fclose(flist);
end
