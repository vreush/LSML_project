import csv

filename = 'train.split.csv'

filename_out = 'train.vw'

# ----- test log loss: 0.393964

def tr(col_name):
	sp = col_name.split('_')
	if len(sp) == 1:
		return col_name
	else:
		return '{}_{}'.format(sp[0][:1], sp[1][:2])


def conv(filename, filename_out, header, is_first=False):
	out = open(filename_out,'w')
	k = 0
	with open(filename, 'rb') as f:
	    reader = csv.reader(f)
	    for row in reader:
	    	if not is_first:
	    		line_out = '{},{}\n'.format(int(row[1]),
	    			','.join(map(lambda x: '{}_{}:1.0'.format(tr(x[0]),x[1]), zip(header[3:], row[3:])))) 
		    	out.write(line_out)
	    	k += 1
	    	is_first = False
	    	if k % 1000 == 0:
	    		print k
	    	# if k >= 1000000:
	    	# 	break
	out.close()


with open(filename, 'rb') as f:
    reader = csv.reader(f)
    header = next(reader)
    print header


print 'train.split.csv'
conv(filename, filename_out, header, is_first=True)
print 'test.split00.csv'
conv('test.split.csv', 'test.vw', header)
