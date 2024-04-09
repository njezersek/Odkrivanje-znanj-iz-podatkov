var fs = require('fs');

fs.readdir('data', function(err, files) {
	files.forEach(function(fileName) {
		clean(fileName)
	});
});

let p0, p1, p2, p3;

function clean(fileName){
	output = "";
	let c = fs.readFileSync(`data/${fileName}`, 'utf8');
	c = c.substring(70, c.length-11);
	o = eval("("+c+")");
	output += "min1800" + "\t" + o.params.p0.name + "\t" + o.params.p1.name + "\t" + o.params.p2.name + "\t" + o.params.p3.name + "\n";
	for(let j in o.points) {
		let points = o.points[j];
		for(let i in points) {
			p0 = points[i].p0 || p0;
			p1 = points[i].p1 || p1;
			p2 = points[i].p2 || p2;
			p3 = points[i].p3 || p3;
			output += i.substring(1) + "\t" + p0 + "\t" + p1 + "\t" + p2 + "\t" + p3 + "\n";
		}
	}
	
	
	// write ouput to file 
	fs.writeFileSync(`data_clean/${fileName}`, output);

	console.log(output);
}