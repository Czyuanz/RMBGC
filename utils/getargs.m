function [eid, emsg, varargout] = getargs(pnames, dflts, varargin)
    %GETARGS Name/value parser reused from CARD-main/utils/getargs.m.
    eid = ''; emsg = '';
    varargout = dflts;
    if mod(numel(varargin),2) ~= 0
        eid='odd_number'; emsg='Optional arguments must be name/value pairs.'; return;
    end
    for j=1:2:numel(varargin)
        name = varargin{j};
        if ~ischar(name) && ~isstring(name)
            eid='bad_name'; emsg='Parameter names must be text.'; return;
        end
        idx = find(strcmpi(char(name), pnames),1);
        if isempty(idx)
            eid='unknown_parameter'; emsg=sprintf('Unknown parameter: %s',char(name)); return;
        end
        varargout{idx}=varargin{j+1};
    end
end
