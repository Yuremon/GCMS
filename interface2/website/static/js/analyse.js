function affiche(){
  var fileSize = 0;
    //get file
    var theFile = document.getElementById("myFile");
    var path = window.document.getElementById("myFile").value;
    var nom=window.document.getElementById("myFile").files[0].name;
    var res = nom.split(".")[0];

     var regex = /^([a-zA-Z0-9\s_\\.\-:])+(.csv|.txt)$/;
     //check if file is CSV
     if (regex.test(theFile.value.toLowerCase())) {
     //check if browser support FileReader
        if (typeof (FileReader) != "undefined") {
       //get table element
        var table = document.getElementById("myTable");
        var headerLine = "";
        //create html5 file reader object
        var myReader = new FileReader();
        alert(res);
      }
    }
    else {
                alert("Please upload a valid CSV file.");
    }
    document.getElementById("affiche").hidden = false;  
    return false;
  }


var previousPath="";

function maFonction(){
var path=window.document.getElementById("monChemin").value;


// permet l'exécution seulement si "path" a été modifié et non null
if(path==previousPath)
  return;
previousPath=path;

// execution
alert(path);
}
