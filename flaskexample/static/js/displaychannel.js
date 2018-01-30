$(document).ready(function () {
  $("div[name=pubdata]").hide();
  $('input[type=radio][name=Radios]').change(function() {
    $("div[name=pubdata]").hide();
    $("div." + this.id).show();
  } );
});
