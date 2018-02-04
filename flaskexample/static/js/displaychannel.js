$(document).ready(function () {
  $("div[name=pubdata]").hide();
  $("div.Radio1").show()
  $('input[type=radio][name=Radios]').change(function() {
    $("div[name=pubdata]").hide();
    $("div." + this.id).show();
  } );
});
