// Custom JavaScript for the NetVLAD project page

$(document).ready(function() {
    // Initialize any carousels or interactive elements
    if (typeof bulmaCarousel !== 'undefined') {
        bulmaCarousel.attach('#results-carousel', {
            slidesToScroll: 1,
            slidesToShow: 1,
            loop: true,
            autoplay: true,
            autoplaySpeed: 3000
        });
    }
    
    // Smooth scrolling for anchor links
    $('a[href*="#"]').on('click', function(e) {
        e.preventDefault();
        var target = $(this.hash);
        if (target.length) {
            $('html, body').animate({
                scrollTop: target.offset().top - 100
            }, 1000);
        }
    });
    
    // Add loading states for images
    $('img').on('load', function() {
        $(this).addClass('loaded');
    });
    
    // Add fade-in animation for sections
    $(window).on('scroll', function() {
        $('.section').each(function() {
            var elementTop = $(this).offset().top;
            var elementBottom = elementTop + $(this).outerHeight();
            var viewportTop = $(window).scrollTop();
            var viewportBottom = viewportTop + $(window).height();
            
            if (elementBottom > viewportTop && elementTop < viewportBottom) {
                $(this).addClass('fade-in');
            }
        });
    });
});
