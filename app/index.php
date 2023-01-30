<html>
<head>
<title>CompositIA</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<script type="text/javascript" src="https://code.jquery.com/jquery-2.1.1.min.js"></script>  
<link rel="stylesheet" href="https://unpkg.com/@picocss/pico@latest/css/pico.min.css">
<style>
h2{
    color: silver;
	margin-bottom: 10px;

}
h4{
    margin-bottom: 10px;
}
.imgpreview img{
    width: 512px;
    height: 512px;
    margin: 10px;
}
.loghi{
	height: 64px;
}
input{
	text-align: left;
}
.loader{
	display: none;
}
.loaderVisible{
	display: inline;
}
.message-hidden{
	display: none;
}
.message-visible{
	display: inline;
	font-size: 10pt;
}
#L3-slice img{
	-webkit-transform: scaleX(-1);
	transform: scaleX(-1);
}
#L3-output img{
	-webkit-transform: scaleX(-1);
	transform: scaleX(-1);	
}
.dotRed {
  height: 16px;
  width: 16px;
  background-color: #FF0000;
  border-radius: 50%;
  display: inline-block;
  margin-left:10px;
}
.dotBlue {
  height: 16px;
  width: 16px;
  background-color: #0000FF;
  border-radius: 50%;
  display: inline-block;
  margin-left:10px;
}
.dotGreen {
  height: 16px;
  width: 16px;
  background-color: #00FF00;
  border-radius: 50%;
  display: inline-block;
  margin-left:10px;
}
.result-title{
	font-weight: bold;
}
#scores-list{
	font-size: 10pt;
}
.score-label{
	width: 250px;
	min-width: 250px;
	display: inline-block;
	border-bottom: dashed black;
	border-width: thin;
}
.score-txt{
	width: 50px;
	min-width: 50px;
	display: inline-block;
	border-bottom: dashed black;
	border-width: thin;
	text-align: right;
}
.div-loghi{
	text-align: right;
}
.container-results-hidden{
	display:none;
}
</style>
<body>

<main class="container">
<div class="grid">
	<div><h2>CompositIA</h2></div>
	<div class="div-loghi"><img class="loghi" src="img/loghi.jpg"/></div>
</div>
<h4>Computation of body composition scores from toraco-abdominal CT scans</h4>
<hr/>
<div class="grid">
	<div>	
		<label for="file" class="input-button">1. Open a NIFTI file</label>
		<input id="file" type="file" onchange="uploadFile(event)"/>
		<img id="loader-upload" class="loader" src="img/loader-horizontal.gif" />
		<span class="message-hidden" id="upload-complete">DONE</span>
	</div>
	<div>
		<label for="init-button" class="predict-button">2. Compute scores</label>
		<input type="button" id="predict-button" class="input-button predict-button" onclick="remoteProcess()" value="Run..."/>
		<img id="loader-process" class="loader" src="img/loader-horizontal.gif" />
		<span class="message-hidden" id="process-complete">DONE</span>
	</div>
</div>
<hr/>
<div id="container-results" class="container-results-hidden">
<div class="grid">
	<div id="sagital-projection">	
		<span class="result-title">Sagital projection</span>
	</div>
	<div id="L1-slice">
		<span class="result-title">Extracted L1 slice</span>
	</div>
	<div id="L3-slice">
		<span class="result-title">Extracted L3 slice</span>
	</div>
</div>

<div class="grid">
	<div id="scores-output">	
		<span class="result-title">Composition scores</span>
		<ul id="scores-list">
			<li><span class="score-label">L1 Spungiosa density avg [Hu]</span><span class="score-txt" id="txtL1DensityAvgHu"></span></li>
			<li><span class="score-label">L1 Spungiosa density std [Hu]</span><span class="score-txt" id="txtL1DensityStdHu"></span></li>
			<li><span class="score-label">L1 Spungiosa area [cm2]</span><span class="score-txt" id="txtL1AreaHu"></span></li>
			<li><span class="score-label">L3 SAT area [cm2]</span><span class="score-txt" id="txtL3SATAreaHu"></span></li>
			<li><span class="score-label">L3 SMA area [cm2]</span><span class="score-txt" id="txtL3SMAAreaHu"></span></li>
			<li><span class="score-label">L3 VAT area [cm2]</span><span class="score-txt" id="txtL3VATAreaHu"></span></li>
			<li><span class="score-label">L3 SAT density std [Hu]</span><span class="score-txt" id="txtL1DensityHu"></span></li>
		</ul>
		
	</div>
	<div id="L1-output">
		<span class="result-title">Segmentation L1</span>
		<span class="dotRed"></span><span>CORT.</span>
		<span class="dotGreen"></span><span>SPUN.</span>
		
	</div>
	<div id="L3-output">
		<span class="result-title">Segmentation L3</span>
		<span class="dotRed"></span><span>SAT</span>
		<span class="dotGreen"></span><span>SMA</span>
		<span class="dotBlue"></span><span>VAT</span>
	</div>
</div>
</div>
</main>
</body>

<script>
var filename = "";
function uploadFile(){
  	$("#loader-upload").addClass("loaderVisible");
	var files = document.getElementById("file").files;

	if(files.length > 0 ){

      var formData = new FormData();
      formData.append("file", files[0]);

      var xhttp = new XMLHttpRequest();

      // Set POST method and ajax file path
      xhttp.open("POST", "ajaxfileupload.php", true);

      // call on request changes state
      xhttp.onreadystatechange = function() {
         if (this.readyState == 4 && this.status == 200) {

           filename = this.responseText;
		   var response = filename.charAt(0);
           if(response == "1"){
			  $("#loader-upload").attr("src", "img/loader-horizontal-done.gif");
			  $("#upload-complete").removeClass("message-hidden");
			  $("#upload-complete").addClass("message-visible");

           }else{
              alert("File not uploaded.");
			  $("#loader-upload").attr("src", "img/loader-horizontal-error.gif");
           }
         }
		 else{
			 $("#loader-upload").attr("src", "img/loader-horizontal-error.gif");
		 }
      };
      // Send request with data
      xhttp.send(formData);

   }else{
      alert("Please select a file");
   }
}

function renderImages(){
	basefn = filename.slice(0, -7);
	var img = $('<img>');
	img.attr('src', 'upload/'+basefn+'/windows/L1slice.png');
	img.appendTo('#L1-slice');
	
	var img = $('<img>');
	img.attr('src', 'upload/'+basefn+'/windows/L3slice.png');
	img.appendTo('#L3-slice');
	
	var img = $('<img>');
	img.attr('src', 'upload/'+basefn+'/windows/combinedSagital.png');
	img.appendTo('#sagital-projection');
	
	var img = $('<img>');
	img.attr('src', 'upload/'+basefn+'/windows/pred_L1slice.png');
	img.appendTo('#L1-output');
	
	var img = $('<img>');
	img.attr('src', 'upload/'+basefn+'/windows/combinedSATSMAVAT.png');
	img.appendTo('#L3-output');	
}

function parseScores(){
	basefn = filename.slice(0, -7);
	$.get( 'upload/'+basefn+'/windows/scores.txt', function( data ) {
		myArray = data.split(" ");
		$( "#txtL1DensityAvgHu" ).html( myArray[0] );
		$( "#txtL1DensityStdHu" ).html( myArray[1] );
		$( "#txtL1AreaHu" ).html( myArray[2] );
		$( "#txtL3SATAreaHu" ).html( myArray[3] );
		$( "#txtL3SMAAreaHu" ).html( myArray[4] );
		$( "#txtL3VATAreaHu" ).html( myArray[5] );
		$( "#txtL1DensityHu" ).html( myArray[6] );
	});
}

function remoteProcess(){
	  $("#loader-process").addClass("loaderVisible");
	  var xhttp = new XMLHttpRequest();
      xhttp.open("GET", "ajaxprocess.php?fn="+filename, true);

      // call on request changes state
      xhttp.onreadystatechange = function() {
         if (this.readyState == 4 && this.status == 200) {

           var process_response = this.responseText;
           if(1){//process_response == "1"
			  $("#loader-process").attr("src", "img/loader-horizontal-done.gif");
			  $("#process-complete").removeClass("message-hidden");
			  $("#container-results").removeClass("container-results-hidden");
			  $("#process-complete").addClass("message-visible");
			  renderImages();
			  parseScores();
           }else{
			  $("#loader-process").attr("src", "img/loader-horizontal-error.gif");
           }
         }
		 else{
			 $("#loader-process").attr("src", "img/loader-horizontal-error.gif");
		 }
      };
      // Send request with data
      xhttp.send();
}

function renderImage(file) {
  var reader = new FileReader();
  console.log("image is here..");
  reader.onload = function(event) {
    img_url = event.target.result;
    console.log("image is here2..");
    document.getElementById("test-image").src = img_url;
  }
  reader.readAsDataURL(file);
}

</script>
</html>
