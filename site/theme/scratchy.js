// Put a link back to the landing page in mdBook's menu bar, so the book and the
// front page read as one site. mdBook has no config hook for this.
(function () {
    const left = document.querySelector('#menu-bar .left-buttons');
    if (!left || document.getElementById('scratchy-home')) {
        return;
    }
    const home = document.createElement('a');
    home.id = 'scratchy-home';
    // Chapters live one level below the site root, spyre/ two.
    home.href = '../'.repeat(window.location.pathname.split('/book/')[1]?.split('/').length || 1);
    home.textContent = 'scratchy';
    home.title = 'Back to the scratchy home page';
    left.appendChild(home);
})();
