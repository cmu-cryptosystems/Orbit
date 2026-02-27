def get_naf_weight(x:int, N:int):
    sign = (x<0)
    x=abs(x)
    sum=0
    for i in range(40):
        if x == 0:
            break
        zi = (2-(x&3)) if (x&1) else 0
        x = ((x-zi)>>1)
        if zi:
            val = round(zi * (1<<i))
            if sign:
                val = -val
            if val%N != 0:
                sum += 1
    
    return sum