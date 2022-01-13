
function FEA_POent() {
		var ptag2 = document.querySelectorAll('FEA_POent');
        var ptag = document.querySelectorAll("FEA_PO");
        var re = /(rgba\(\d+\, \d+\, \d+\, )0(\))/;
        var re2 = /rgb(\(\d+\, \d+\, \d+)(\))/;

    for(var i=0;i<ptag2.length;i++){

      if (ptag2[i].style.backgroundColor == "rgba(239, 186, 214, 0)") {
        var bc = ptag2[i].style.backgroundColor;
      	var newbc = bc.replace(re, '$11$2');
        ptag2[i].style.backgroundColor=newbc;
        ptag[i].style.display = 'inline';
      } else {
        var bc = ptag2[i].style.backgroundColor;
      	var orgbc = bc.replace(re2, 'rgba$1\, 0$2');
        ptag2[i].style.backgroundColor=orgbc;
        ptag[i].style.display = 'none';
      }
    }
	}
    
function FEA_SPent() {
		var ptag2 = document.querySelectorAll('FEA_SPent');
        var ptag = document.querySelectorAll("FEA_SP");
        var re = /(rgba\(\d+\, \d+\, \d+\, )0(\))/;
        var re2 = /rgb(\(\d+\, \d+\, \d+)(\))/;

    for(var i=0;i<ptag2.length;i++){

      if (ptag2[i].style.backgroundColor == "rgba(255, 214, 170, 0)") {
        var bc = ptag2[i].style.backgroundColor;
      	var newbc = bc.replace(re, '$11$2');
        ptag2[i].style.backgroundColor=newbc;
        ptag[i].style.display = 'inline';
      } else {
        var bc = ptag2[i].style.backgroundColor;
      	var orgbc = bc.replace(re2, 'rgba$1\, 0$2');
        ptag2[i].style.backgroundColor=orgbc;
        ptag[i].style.display = 'none';
      }
    }
	}
    
function FEA_WPent() {
		var ptag2 = document.querySelectorAll('FEA_WPent');
        var ptag = document.querySelectorAll("FEA_WP");
        var re = /(rgba\(\d+\, \d+\, \d+\, )0(\))/;
        var re2 = /rgb(\(\d+\, \d+\, \d+)(\))/;

    for(var i=0;i<ptag2.length;i++){

      if (ptag2[i].style.backgroundColor == "rgba(199, 214, 216, 0)") {
        var bc = ptag2[i].style.backgroundColor;
      	var newbc = bc.replace(re, '$11$2');
        ptag2[i].style.backgroundColor=newbc;
        ptag[i].style.display = 'inline';
      } else {
        var bc = ptag2[i].style.backgroundColor;
      	var orgbc = bc.replace(re2, 'rgba$1\, 0$2');
        ptag2[i].style.backgroundColor=orgbc;
        ptag[i].style.display = 'none';
      }
    }
	}
    
function FEA_NGent() {
		var ptag2 = document.querySelectorAll('FEA_NGent');
        var ptag = document.querySelectorAll("FEA_NG");
        var re = /(rgba\(\d+\, \d+\, \d+\, )0(\))/;
        var re2 = /rgb(\(\d+\, \d+\, \d+)(\))/;

    for(var i=0;i<ptag2.length;i++){

      if (ptag2[i].style.backgroundColor == "rgba(206, 132, 103, 0)") {
        var bc = ptag2[i].style.backgroundColor;
      	var newbc = bc.replace(re, '$11$2');
        ptag2[i].style.backgroundColor=newbc;
        ptag[i].style.display = 'inline';
      } else {
        var bc = ptag2[i].style.backgroundColor;
      	var orgbc = bc.replace(re2, 'rgba$1\, 0$2');
        ptag2[i].style.backgroundColor=orgbc;
        ptag[i].style.display = 'none';
      }
    }
	}
    
function FEA_SNent() {
		var ptag2 = document.querySelectorAll('FEA_SNent');
        var ptag = document.querySelectorAll("FEA_SN");
        var re = /(rgba\(\d+\, \d+\, \d+\, )0(\))/;
        var re2 = /rgb(\(\d+\, \d+\, \d+)(\))/;

    for(var i=0;i<ptag2.length;i++){

      if (ptag2[i].style.backgroundColor == "rgba(228, 169, 155, 0)") {
        var bc = ptag2[i].style.backgroundColor;
      	var newbc = bc.replace(re, '$11$2');
        ptag2[i].style.backgroundColor=newbc;
        ptag[i].style.display = 'inline';
      } else {
        var bc = ptag2[i].style.backgroundColor;
      	var orgbc = bc.replace(re2, 'rgba$1\, 0$2');
        ptag2[i].style.backgroundColor=orgbc;
        ptag[i].style.display = 'none';
      }
    }
	}
    
function FEA_WNent() {
		var ptag2 = document.querySelectorAll('FEA_WNent');
        var ptag = document.querySelectorAll("FEA_WN");
        var re = /(rgba\(\d+\, \d+\, \d+\, )0(\))/;
        var re2 = /rgb(\(\d+\, \d+\, \d+)(\))/;

    for(var i=0;i<ptag2.length;i++){

      if (ptag2[i].style.backgroundColor == "rgba(131, 139, 178, 0)") {
        var bc = ptag2[i].style.backgroundColor;
      	var newbc = bc.replace(re, '$11$2');
        ptag2[i].style.backgroundColor=newbc;
        ptag[i].style.display = 'inline';
      } else {
        var bc = ptag2[i].style.backgroundColor;
      	var orgbc = bc.replace(re2, 'rgba$1\, 0$2');
        ptag2[i].style.backgroundColor=orgbc;
        ptag[i].style.display = 'none';
      }
    }
	}
    function getCheckboxValue(event) {
        var ptag2 = document.querySelectorAll('.entity');
        var ptag = document.querySelectorAll(".tagset");
        
        var re = /(rgba\(\d+\, \d+\, \d+\, )0(\))/;
        var re2 = /rgb(\(\d+\, \d+\, \d+)(\))/;
        
        for(var i=0;i<ptag2.length;i++){

            let result = '';
            if(event.target.checked)  {
            var bc = ptag2[i].style.backgroundColor;
            var newbc = bc.replace(re, '$11$2');
            ptag2[i].style.backgroundColor=newbc;
            ptag[i].style.display = 'inline';
            }else {
            var bc = ptag2[i].style.backgroundColor;
            var orgbc = bc.replace(re2, 'rgba$1\, 0$2');
            ptag2[i].style.backgroundColor=orgbc;
            ptag[i].style.display = 'none';
            }
        }
        }
