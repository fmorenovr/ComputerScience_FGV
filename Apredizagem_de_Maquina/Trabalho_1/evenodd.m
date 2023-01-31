function [r] = evenodd(n, e)
  if e~=1 && e~=2
    return
  end

  v = randi([0, 30],1,n);

  if e==1
    iwant_odd = v(1:2:end);
    r = iwant_odd(mod(iwant_odd,2)==1);
  elseif e==2
    iwant_even = v(2:2:end);
    r = iwant_even(mod(iwant_even,2)==0);
  end
end