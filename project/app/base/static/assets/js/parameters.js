/* function getPath(data) {
    return data
} */

function saveFile(projs,data) {
	
	/*alert(projs);*/
	const nameVal = document.getElementById("project").value;
	const modeVal = document.getElementById("trainingmode").value;
	const algoVal = document.getElementById("algorithm").value;
	const epochsVal = document.getElementById("epochs").value;
	const perVal = document.getElementById("scale2").value;	
	const degVal = document.getElementById("scale3").value;
	const maxVal = document.getElementById("scale4").value;
	let project = { 
		name: nameVal,
		mode: modeVal,
		algorithm: algoVal,
		epochs: epochsVal,
		percentage: perVal,
		max_degree: degVal,
		max_nodes: maxVal
	}; 

	let d = JSON.stringify(project); 
	/* alert(d);*/
	const textToBLOB = new Blob([d], { type: 'text/plain' });
	const sFileName = nameVal+".json";	   // The file to save the data.

	let newLink = document.createElement("a");
	newLink.download = sFileName;

	if (window.webkitURL != null) {
		newLink.href = window.webkitURL.createObjectURL(textToBLOB);
	}
	else {
		newLink.href = window.URL.createObjectURL(textToBLOB);
		newLink.style.display = "none";
		document.body.appendChild(newLink);
	}

	newLink.click();
	/*reset inputs*/
	// document.getElementById("project")="";
	// document.getElementById("prefix")="";
	// document.getElementById("project")="";

} 


/* function clearInput(f){
    if(f.value){
       f.value = ''; //for IE11, latest Chrome/Firefox/Opera...
    }
} */

