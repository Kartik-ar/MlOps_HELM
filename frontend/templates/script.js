async function show(){
    const xhttp = new XMLHttpRequest();
      // Send a request
    xhttp.open("GET", "http://13.234.239.157:30007/preds");
    await xhttp.send();
    xhttp.onload = function() {
        const obj = this.responseText;
        const data = obj.replace(/\'/g, '"');
        var D = JSON.parse(data);
        //var RF_val = D.RF_prediction.values
        //var LR_val = D.LR.values
        var t = "<label> Predicted Values </label> <br> <table class='values'> <tr>\
        <th> Actual Values </th> \
        <th> Random Forest </th> \
        <th> Linear Regression </th>\
        <th> Naive Bayes </th>\
        <th> K-Nearest Neighbour </th>\
        </tr>"
        for (let i = 0, len = Object.keys(D.LR.values).length-1; i < len; i++ ) {
          t += "<tr>" +
          "<td>" + D.actual.values[i] + "</td>" +
          "<td>" + D.RF.values[i] + "</td>" +
          "<td>" + Math.round(D.LR.values[i]) + "</td>"+
          "<td>" + D.NB.values[i] + "</td>"+
          "<td>" + D.KNN.values[i] + "</td>"+
          "</tr>"
        };
        /*for (let i = 0 ; i < Object.keys(D.LR.values).length-1; i++){
        console.log(D.RF.values[i])
        }*/
        console.log(t + "</table> <br> <button onclick='bt()'> Back </button>")
        //alert(t + RF + "</table> <br> <button class='button' onclick='bt()'> Back </button>")
        document.getElementById('overlay').innerHTML = t + "</table> <br> <button class='button' onclick='bt()'> Back </button>";
        //Object.keys(D.LR.values) );
      };
}

async function score(){
  const xhttp = new XMLHttpRequest();
    // Send a request
  xhttp.open("GET", "http://13.234.239.157:30007/preds");
  await xhttp.send();
  xhttp.onload = function() {
      const obj = this.responseText;
      const data = obj.replace(/\'/g, '"');
      var D = JSON.parse(data);
      //var RF_val = D.RF_prediction.values
      //var LR_val = D.LR.values
      var t = "<label> Scores Of all Models </label> <br> <table class='values'> <tr>\
      <th> Random Forest </th> \
      <th> Linear Regression </th>\
      <th> Naive Bayes </th>\
      <th> K-Nearest Neighbour </th>\
      </tr>"
      //for (let i = 0, len = Object.keys(D.LR.score).length-1; i < len; i++ ) {
      t += "<tr>" +
        "<td>" + D.RF.score + "</td>" +
        "<td>" + D.LR.score + "</td>"+
        "<td>" + D.NB.score + "</td>"+
        "<td>" + D.KNN.score + "</td>"+
        "</tr>"
      document.getElementById('overlay').innerHTML = t + "</table> <br> <button class='button' onclick='bt()'> Back </button>";
      
    }
  }

  async function mse(){
    const xhttp = new XMLHttpRequest();
      // Send a request
    xhttp.open("GET", "http://13.234.239.157:30007/preds");
    await xhttp.send();
    xhttp.onload = function() {
        const obj = this.responseText;
        const data = obj.replace(/\'/g, '"');
        var D = JSON.parse(data);
        //var RF_val = D.RF_prediction.values
        //var LR_val = D.LR.values
        var t = "<label> Mean Square Error for all Models </label> <br> <table class='values'> <tr>\
        <th> Random Forest </th> \
        <th> Linear Regression </th>\
        <th> Naive Bayes </th>\
        <th> K-Nearest Neighbour </th>\
        </tr>"
        //for (let i = 0, len = Object.keys(D.LR.score).length-1; i < len; i++ ) {
        t += "<tr>" +
          "<td>" + D.RF.MSE + "</td>" +
          "<td>" + D.LR.MSE + "</td>"+
          "<td>" + D.NB.MSE + "</td>"+
          "<td>" + D.KNN.MSE + "</td>"+
          "</tr>"
        document.getElementById('overlay').innerHTML = t + "</table> <br> <button class='button' onclick='bt()'> Back </button>";
        
      }
    }

function bt(){
  var x = "<table> \
  <button class='button' onclick='show()' >Compare Predicted Value of Test Dataset</button> <br>\
  <button class='button' onclick='score()'>Compare Scores</button><br>\
  <button class='button' onclick='mse()'> Compare with Mean Sauared Errors</button>\
  </table>"
  document.getElementById('overlay').innerHTML = x;
}
