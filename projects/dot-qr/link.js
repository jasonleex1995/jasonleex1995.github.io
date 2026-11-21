/* Dot QR — take the tracking off a link.

   Links copied from apps usually end in parameters that only record where the click came from
   (?utm_source=share&utm_medium=ios_app…, ?igsh=…, ?si=…). They don't change the page, but they make
   the link — and so the QR code — much longer: a LinkedIn profile shared from the app is about 120
   characters, 41 without them. cleanLink removes those, and nothing else: the address, the other
   parameters (in their order and spelling) and the #fragment stay exactly as typed.

   The page loads it as window.DotLink; check.js require()s it.
*/
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.DotLink = factory();
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // Tracking parameters on any site: campaign tags and ad-click ids.
  var ANY = /^(utm_[a-z0-9_]+|fbclid|gclid|gclsrc|dclid|gbraid|wbraid|msclkid|mc_cid|mc_eid|igsh|igshid|ttclid|twclid|yclid|_gl)$/i;
  // Site by site: share ids that only matter there. (YouTube's t= is a time and stays; X's t= is tracking.)
  var SITES = [
    [/(^|\.)(youtube\.com|youtu\.be)$/, /^(si|feature|pp)$/i],
    [/(^|\.)(x\.com|twitter\.com)$/, /^(s|t|ref_src|ref_url)$/i],
    [/(^|\.)linkedin\.com$/, /^(trk|trkinfo|trackingid|lipi|originalsubdomain)$/i],
    [/(^|\.)spotify\.com$/, /^(si)$/i],
    [/(^|\.)threads\.(net|com)$/, /^(xmt|slof)$/i],
    [/(^|\.)facebook\.com$/, /^(mibextid)$/i],
    [/(^|\.)tiktok\.com$/, /^(is_from_webapp|sender_device)$/i]
  ];

  function keyOf(pair) {
    var k = pair.split('=')[0];
    try { return decodeURIComponent(k.replace(/\+/g, ' ')); } catch (e) { return k; }
  }

  // → { link, removed: [parameter names] }. A link with nothing to remove comes back unchanged.
  function cleanLink(link) {
    var m = /^([a-z][a-z0-9+.\-]*:\/\/([^\/?#]*))([^?#]*)(\?[^#]*)?(#[\s\S]*)?$/i.exec(link || '');
    if (!m || !m[4]) return { link: link, removed: [] };
    var host = m[2].replace(/^[^@]*@/, '').replace(/:\d*$/, '').toLowerCase(), site = null;
    for (var i = 0; i < SITES.length; i++) if (SITES[i][0].test(host)) site = SITES[i][1];
    var kept = [], removed = [];
    m[4].slice(1).split('&').forEach(function (pair) {
      if (!pair) return;
      var key = keyOf(pair);
      if (ANY.test(key) || (site && site.test(key))) removed.push(key);
      else kept.push(pair);
    });
    if (!removed.length) return { link: link, removed: [] };
    return { link: m[1] + m[3] + (kept.length ? '?' + kept.join('&') : '') + (m[5] || ''), removed: removed };
  }

  return { cleanLink: cleanLink };
}));
