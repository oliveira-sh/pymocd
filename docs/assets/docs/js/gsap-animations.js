/**
 * GSAP Animations Orchestrator for PyMOCD Documentation
 * Playful & Energetic animation suite using GSAP and ScrollTrigger
 */

// Register GSAP plugins
gsap.registerPlugin(ScrollTrigger);

// ===========================================
// UTILITY FUNCTIONS
// ===========================================

/**
 * Check if element exists
 */
function exists(selector) {
    return document.querySelector(selector) !== null;
}

/**
 * Get all elements
 */
function getAll(selector) {
    return document.querySelectorAll(selector);
}

// ===========================================
// HERO SECTION ANIMATIONS
// ===========================================

function animateHero() {
    if (!exists('.hero')) return;

    const tl = gsap.timeline({ defaults: { ease: 'back.out(1.7)' } });

    // Badge pop-in with elastic bounce
    if (exists('.hero .badge')) {
        tl.from('.hero .badge', {
            y: -100,
            opacity: 0,
            scale: 0,
            duration: 0.8,
            ease: 'elastic.out(1, 0.5)'
        }, 0.2);
    }

    // Title: slide-up with fade (simpler animation to avoid DOM manipulation)
    if (exists('.hero h1, .hero .hero-title')) {
        tl.from('.hero h1, .hero .hero-title', {
            y: 50,
            opacity: 0,
            duration: 0.8,
            ease: 'back.out(1.7)'
        }, 0.5);
    }

    // Subtitle fade-in delayed
    if (exists('.hero p, .hero .lead')) {
        tl.from('.hero p, .hero .lead', {
            y: 30,
            opacity: 0,
            duration: 0.8
        }, 1.0);
    }

    // CTA buttons: bounce in with breathing animation
    if (exists('.hero .btn')) {
        tl.from('.hero .btn', {
            scale: 0,
            opacity: 0,
            duration: 0.6,
            stagger: 0.15,
            ease: 'back.out(1.7)',
            clearProps: 'all'
        }, 1.2).set('.hero .btn', {
            opacity: 1
        });

        // Add breathing pulse animation (only scale, not opacity)
        gsap.fromTo('.hero .btn',
            {
                scale: 1
            },
            {
                scale: 1.0,
                duration: 1.5,
                repeat: -1,
                yoyo: true,
                ease: 'sine.inOut',
                delay: 1.8
            }
        );
    }

    // Floating gradient orbs background (if container exists)
    if (exists('.hero')) {
        createFloatingOrbs('.hero');
    }
}

/**
 * Create floating orb animations in background
 */
function createFloatingOrbs(container) {
    const containerEl = document.querySelector(container);
    if (!containerEl) return;

    for (let i = 0; i < 3; i++) {
        const orb = document.createElement('div');
        orb.className = 'floating-orb';
        orb.style.cssText = `
            position: absolute;
            border-radius: 50%;
            filter: blur(60px);
            opacity: 0.15;
            pointer-events: none;
            z-index: 0;
        `;

        const size = gsap.utils.random(200, 400);
        const colors = ['#4F46E5', '#06B6D4', '#8B5CF6', '#EC4899'];

        gsap.set(orb, {
            width: size,
            height: size,
            background: colors[i % colors.length],
            x: gsap.utils.random(0, containerEl.offsetWidth - size),
            y: gsap.utils.random(0, containerEl.offsetHeight - size)
        });

        containerEl.appendChild(orb);

        // Floating animation
        gsap.to(orb, {
            x: `+=${gsap.utils.random(-200, 200)}`,
            y: `+=${gsap.utils.random(-200, 200)}`,
            duration: gsap.utils.random(8, 15),
            repeat: -1,
            yoyo: true,
            ease: 'sine.inOut'
        });
    }
}

// ===========================================
// FEATURE GRID ANIMATIONS
// ===========================================

function animateFeatureGrid() {
    const features = getAll('.feature-card, .card.feature, .features, .key-feature');
    if (features.length === 0) return;

    features.forEach((card, index) => {
        // Initial state
        gsap.set(card, { opacity: 0, y: 50, scale: 0.9 });

        // Scroll-triggered entrance with stagger
        ScrollTrigger.create({
            trigger: card,
            start: 'top 85%',
            onEnter: () => {
                gsap.to(card, {
                    opacity: 1,
                    y: 0,
                    scale: 1,
                    duration: 0.8,
                    delay: index * 0.1,
                    ease: 'elastic.out(1, 0.75)'
                });
            },
            once: true
        });

        // Icon spin on card hover
        const icon = card.querySelector('.material-icons, i, svg');
        if (icon) {
            card.addEventListener('mouseenter', () => {
                gsap.to(icon, {
                    rotation: 360,
                    duration: 0.6,
                    ease: 'back.out(1.7)'
                });
            });
        }
    });

    // Add 3D tilt effect to cards
    add3DTiltEffect('.feature-card, .card.feature, .features, .key-feature');
}

/**
 * Add 3D tilt effect on mouse move
 */
function add3DTiltEffect(selector) {
    const cards = getAll(selector);

    cards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const centerX = rect.width / 2;
            const centerY = rect.height / 2;

            const rotateX = (y - centerY) / 10;
            const rotateY = (centerX - x) / 10;

            gsap.to(card, {
                rotationX: rotateX,
                rotationY: rotateY,
                transformPerspective: 1000,
                duration: 0.3,
                ease: 'power2.out'
            });
        });

        card.addEventListener('mouseleave', () => {
            gsap.to(card, {
                rotationX: 0,
                rotationY: 0,
                duration: 0.5,
                ease: 'elastic.out(1, 0.5)'
            });
        });
    });
}

// ===========================================
// LANDING PAGE SECTIONS
// ===========================================

function animateLandingSections() {
    // Animate section headings (h2, h3)
    const sectionHeadings = getAll('section h2, section h3, .section-title');
    sectionHeadings.forEach((heading, index) => {
        gsap.from(heading, {
            scrollTrigger: {
                trigger: heading,
                start: 'top 85%'
            },
            y: 50,
            opacity: 0,
            duration: 0.8,
            ease: 'back.out(1.7)'
        });
    });

    // Animate section descriptions/paragraphs
    const sectionDescriptions = getAll('section > p, section .lead, .section-description');
    sectionDescriptions.forEach((desc, index) => {
        gsap.from(desc, {
            scrollTrigger: {
                trigger: desc,
                start: 'top 88%'
            },
            y: 30,
            opacity: 0,
            duration: 0.6,
            delay: 0.2,
            ease: 'power2.out'
        });
    });
}

// ===========================================
// IMAGE COMPARE SECTION
// ===========================================

function animateImageCompare() {
    const sections = getAll('.image-compare-section, .image-text');

    sections.forEach((section, index) => {
        const isEven = index % 2 === 0;
        const direction = isEven ? -100 : 100;

        gsap.from(section, {
            scrollTrigger: {
                trigger: section,
                start: 'top 80%'
            },
            x: direction,
            opacity: 0,
            duration: 1,
            ease: 'back.out(1.4)'
        });
    });
}

// ===========================================
// NAVIGATION ANIMATIONS
// ===========================================

/**
 * Landing page header animations
 */
function animateLandingHeader() {
    if (!exists('header .navbar, .topbar')) return;

    const tl = gsap.timeline();

    // Logo bounce-in
    if (exists('.navbar-brand, .logo')) {
        tl.from('.navbar-brand, .logo', {
            scale: 0,
            rotation: -180,
            duration: 0.8,
            ease: 'back.out(1.7)'
        }, 0);
    }

    // Nav items staggered fade-in
    const navItems = getAll('.navbar-nav .nav-link, .nav-menu a');
    if (navItems.length > 0) {
        tl.from(navItems, {
            y: -20,
            opacity: 0,
            stagger: 0.1,
            duration: 0.5,
            ease: 'back.out(1.7)'
        }, 0.3);
    }

    // Underline animation on hover
    addUnderlineAnimation('.navbar-nav .nav-link, .nav-menu a');
}

/**
 * Add underline sweep animation
 */
function addUnderlineAnimation(selector) {
    const links = getAll(selector);

    links.forEach(link => {
        // Create underline element
        const underline = document.createElement('span');
        underline.style.cssText = `
            position: absolute;
            bottom: 0;
            left: 0;
            width: 0;
            height: 2px;
            background: currentColor;
            transition: width 0.3s ease;
        `;

        if (link.style.position !== 'absolute') {
            link.style.position = 'relative';
        }
        link.appendChild(underline);

        link.addEventListener('mouseenter', () => {
            gsap.to(underline, {
                width: '100%',
                duration: 0.4,
                ease: 'power2.out'
            });
        });

        link.addEventListener('mouseleave', () => {
            gsap.to(underline, {
                width: 0,
                duration: 0.4,
                ease: 'power2.in'
            });
        });
    });
}

/**
 * Sidebar accordion animations
 */
function animateSidebar() {
    const sidebarLinks = getAll('.sidebar-nav .nav-link, .docs-links .nav-link');

    sidebarLinks.forEach(link => {
        const hasChildren = link.nextElementSibling && link.nextElementSibling.classList.contains('nav');

        if (hasChildren) {
            const chevron = link.querySelector('.dropdown-chevron, .material-icons');
            const submenu = link.nextElementSibling;

            link.addEventListener('click', (e) => {
                const isExpanded = link.getAttribute('aria-expanded') === 'true';

                // Rotate chevron with bounce
                if (chevron) {
                    gsap.to(chevron, {
                        rotation: isExpanded ? 0 : 90,
                        duration: 0.5,
                        ease: 'back.out(1.7)'
                    });
                }

                // Animate submenu height
                if (submenu && !isExpanded) {
                    gsap.from(submenu.children, {
                        x: -20,
                        opacity: 0,
                        stagger: 0.05,
                        duration: 0.4,
                        ease: 'back.out(1.4)'
                    });
                }
            });
        }
    });

    // Active item sliding highlight
    const activeLink = document.querySelector('.sidebar-nav .active, .docs-links .active');
    if (activeLink) {
        const highlight = document.createElement('div');
        highlight.className = 'active-highlight';
        highlight.style.cssText = `
            position: absolute;
            left: 0;
            width: 3px;
            height: 100%;
            background: var(--primary-color, #4F46E5);
            border-radius: 0 2px 2px 0;
            transition: top 0.3s ease, height 0.3s ease;
        `;

        activeLink.style.position = 'relative';
        activeLink.appendChild(highlight);

        // Pulse animation
        gsap.to(highlight, {
            opacity: 0.6,
            duration: 1,
            repeat: -1,
            yoyo: true,
            ease: 'sine.inOut'
        });
    }
}

/**
 * Table of Contents animations
 */
function animateTOC() {
    const tocLinks = getAll('#TableOfContents a, .toc a');
    if (tocLinks.length === 0) return;

    // Create sliding indicator
    const toc = document.querySelector('#TableOfContents, .toc');
    if (toc) {
        const indicator = document.createElement('div');
        indicator.className = 'toc-indicator';
        indicator.style.cssText = `
            position: absolute;
            left: 0;
            width: 2px;
            height: 20px;
            background: linear-gradient(to bottom, #4F46E5, #06B6D4);
            border-radius: 2px;
            transition: top 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            pointer-events: none;
        `;
        toc.style.position = 'relative';
        toc.appendChild(indicator);
    }

    // Smooth scroll on click
    tocLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href');
            const target = document.querySelector(targetId);

            if (target) {
                gsap.to(window, {
                    scrollTo: {
                        y: target,
                        offsetY: 80
                    },
                    duration: 1,
                    ease: 'power2.inOut'
                });
            }
        });

        // Wiggle on hover
        link.addEventListener('mouseenter', () => {
            gsap.to(link, {
                x: 5,
                duration: 0.3,
                ease: 'back.out(2)'
            });
        });

        link.addEventListener('mouseleave', () => {
            gsap.to(link, {
                x: 0,
                duration: 0.3,
                ease: 'back.out(2)'
            });
        });
    });
}

/**
 * Breadcrumbs animations
 */
function animateBreadcrumbs() {
    const breadcrumbs = document.querySelector('.breadcrumb, nav[aria-label="breadcrumb"]');
    if (!breadcrumbs) return;

    // Fade-in slide-up on page load
    gsap.from(breadcrumbs, {
        y: -20,
        opacity: 0,
        duration: 0.6,
        ease: 'back.out(1.7)'
    });

    // Chevron wiggle on hover
    const separators = getAll('.breadcrumb .material-icons');
    separators.forEach(icon => {
        const parent = icon.parentElement;
        parent.addEventListener('mouseenter', () => {
            gsap.to(icon, {
                x: 3,
                duration: 0.2,
                repeat: 3,
                yoyo: true,
                ease: 'power1.inOut'
            });
        });
    });
}

// ===========================================
// CONTENT REVEAL ANIMATIONS
// ===========================================

function animateContent() {
    // Headings slide-in from left
    const headings = getAll('.docs-content h1, .docs-content h2, .docs-content h3, .main-content h1, .main-content h2, .main-content h3');
    headings.forEach((heading, index) => {
        gsap.from(heading, {
            scrollTrigger: {
                trigger: heading,
                start: 'top 90%'
            },
            x: -50,
            opacity: 0,
            duration: 0.8,
            delay: index * 0.05,
            ease: 'back.out(1.4)'
        });
    });

    // Paragraphs fade-in
    const paragraphs = getAll('.docs-content p, .main-content p');
    paragraphs.forEach((p, index) => {
        gsap.from(p, {
            scrollTrigger: {
                trigger: p,
                start: 'top 92%'
            },
            opacity: 0,
            y: 20,
            duration: 0.6,
            delay: index * 0.02,
            ease: 'power2.out'
        });
    });

    // Code blocks reveal with scale
    const codeBlocks = getAll('pre, .highlight');
    codeBlocks.forEach(block => {
        gsap.from(block, {
            scrollTrigger: {
                trigger: block,
                start: 'top 85%'
            },
            scale: 0.95,
            opacity: 0,
            duration: 0.6,
            ease: 'back.out(1.4)'
        });
    });

    // Images zoom-in with blur
    const images = getAll('.docs-content img, .main-content img');
    images.forEach(img => {
        gsap.from(img, {
            scrollTrigger: {
                trigger: img,
                start: 'top 85%'
            },
            scale: 0.8,
            opacity: 0,
            filter: 'blur(10px)',
            duration: 0.8,
            ease: 'power2.out'
        });
    });

    // List items sequential pop-in
    const lists = getAll('.docs-content ul, .docs-content ol, .main-content ul, .main-content ol');
    lists.forEach(list => {
        const items = list.querySelectorAll('li');
        gsap.from(items, {
            scrollTrigger: {
                trigger: list,
                start: 'top 88%'
            },
            scale: 0.8,
            opacity: 0,
            x: -20,
            stagger: 0.1,
            duration: 0.5,
            ease: 'back.out(1.7)'
        });
    });
}

// ===========================================
// CARD & CALLOUT ANIMATIONS
// ===========================================

function animateCards() {
    const cards = getAll('.card:not(.feature-card)');

    cards.forEach(card => {
        gsap.from(card, {
            scrollTrigger: {
                trigger: card,
                start: 'top 85%'
            },
            y: 30,
            opacity: 0,
            scale: 0.95,
            duration: 0.6,
            ease: 'back.out(1.4)'
        });
    });

    // Alert boxes slide-down
    const alerts = getAll('.alert');
    alerts.forEach(alert => {
        const icon = alert.querySelector('.material-icons, i, svg');

        gsap.from(alert, {
            scrollTrigger: {
                trigger: alert,
                start: 'top 90%'
            },
            y: -20,
            opacity: 0,
            duration: 0.5,
            ease: 'back.out(1.7)'
        });

        // Icon spin
        if (icon) {
            gsap.from(icon, {
                scrollTrigger: {
                    trigger: alert,
                    start: 'top 90%'
                },
                rotation: -180,
                scale: 0,
                duration: 0.6,
                delay: 0.2,
                ease: 'back.out(1.7)'
            });
        }
    });
}

// ===========================================
// BUTTON ANIMATIONS
// ===========================================

function animateButtons() {
    const buttons = getAll('.btn:not(.hero .btn)');

    buttons.forEach(btn => {
        // Ripple effect on click
        btn.addEventListener('click', function(e) {
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;

            ripple.style.cssText = `
                position: absolute;
                width: ${size}px;
                height: ${size}px;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.6);
                top: ${y}px;
                left: ${x}px;
                pointer-events: none;
                transform: scale(0);
            `;

            this.style.position = 'relative';
            this.style.overflow = 'hidden';
            this.appendChild(ripple);

            gsap.to(ripple, {
                scale: 2,
                opacity: 0,
                duration: 0.6,
                ease: 'power2.out',
                onComplete: () => ripple.remove()
            });
        });

        // Icon slide-in on hover
        const icon = btn.querySelector('.material-icons, i, svg');
        if (icon) {
            const originalX = icon.offsetLeft;

            btn.addEventListener('mouseenter', () => {
                gsap.from(icon, {
                    x: -10,
                    opacity: 0,
                    duration: 0.3,
                    ease: 'back.out(2)'
                });
            });
        }
    });
}

// ===========================================
// DOC NAVIGATION ANIMATIONS
// ===========================================

function animateDocNav() {
    const prevCard = document.querySelector('.doc-nav-prev, .docs-nav-prev');
    const nextCard = document.querySelector('.doc-nav-next, .docs-nav-next');

    if (prevCard) {
        gsap.from(prevCard, {
            scrollTrigger: {
                trigger: prevCard,
                start: 'top 90%'
            },
            x: -50,
            opacity: 0,
            duration: 0.8,
            ease: 'back.out(1.4)'
        });

        // Arrow bounce on hover
        const arrow = prevCard.querySelector('.material-icons, i, svg');
        if (arrow) {
            prevCard.addEventListener('mouseenter', () => {
                gsap.to(arrow, {
                    x: -5,
                    duration: 0.3,
                    repeat: 2,
                    yoyo: true,
                    ease: 'power1.inOut'
                });
            });
        }
    }

    if (nextCard) {
        gsap.from(nextCard, {
            scrollTrigger: {
                trigger: nextCard,
                start: 'top 90%'
            },
            x: 50,
            opacity: 0,
            duration: 0.8,
            ease: 'back.out(1.4)'
        });

        // Arrow bounce on hover
        const arrow = nextCard.querySelector('.material-icons, i, svg');
        if (arrow) {
            nextCard.addEventListener('mouseenter', () => {
                gsap.to(arrow, {
                    x: 5,
                    duration: 0.3,
                    repeat: 2,
                    yoyo: true,
                    ease: 'power1.inOut'
                });
            });
        }
    }
}

// ===========================================
// TABS ANIMATIONS
// ===========================================

function animateTabs() {
    const tabGroups = getAll('[role="tablist"]');

    tabGroups.forEach(tabList => {
        const tabs = tabList.querySelectorAll('[role="tab"]');
        const indicator = document.createElement('div');

        indicator.style.cssText = `
            position: absolute;
            bottom: 0;
            height: 2px;
            background: var(--primary-color, #4F46E5);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            pointer-events: none;
        `;

        tabList.style.position = 'relative';
        tabList.appendChild(indicator);

        // Position indicator under active tab
        const activeTab = tabList.querySelector('[aria-selected="true"]');
        if (activeTab) {
            indicator.style.left = activeTab.offsetLeft + 'px';
            indicator.style.width = activeTab.offsetWidth + 'px';
        }

        tabs.forEach(tab => {
            tab.addEventListener('click', function() {
                gsap.to(indicator, {
                    left: this.offsetLeft,
                    width: this.offsetWidth,
                    duration: 0.3,
                    ease: 'power2.out'
                });
            });
        });
    });

    // Tab panel content crossfade
    const tabPanels = getAll('[role="tabpanel"]');
    tabPanels.forEach(panel => {
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.attributeName === 'hidden' && !panel.hidden) {
                    gsap.from(panel, {
                        opacity: 0,
                        y: 10,
                        duration: 0.4,
                        ease: 'power2.out'
                    });
                }
            });
        });

        observer.observe(panel, { attributes: true });
    });
}

// ===========================================
// BACK TO TOP BUTTON
// ===========================================

function animateBackToTop() {
    const btn = document.querySelector('#back-to-top, .back-to-top');
    if (!btn) return;

    // Initially hide
    gsap.set(btn, { scale: 0, opacity: 0 });

    // Show/hide on scroll with bounce
    ScrollTrigger.create({
        start: 'top -500',
        end: 'max',
        onEnter: () => {
            gsap.to(btn, {
                scale: 1,
                opacity: 1,
                duration: 0.5,
                ease: 'elastic.out(1, 0.5)'
            });
        },
        onLeaveBack: () => {
            gsap.to(btn, {
                scale: 0,
                opacity: 0,
                duration: 0.3,
                ease: 'power2.in'
            });
        }
    });

    // Bounce on hover
    btn.addEventListener('mouseenter', () => {
        gsap.to(btn, {
            scale: 1.1,
            rotation: -10,
            duration: 0.3,
            ease: 'back.out(2)'
        });
    });

    btn.addEventListener('mouseleave', () => {
        gsap.to(btn, {
            scale: 1,
            rotation: 0,
            duration: 0.3,
            ease: 'back.out(2)'
        });
    });
}

// ===========================================
// DARK MODE TOGGLE ANIMATION
// ===========================================

function animateDarkModeToggle() {
    const toggle = document.querySelector('#darkModeSwitch, .dark-mode-toggle');
    if (!toggle) return;

    toggle.addEventListener('change', function() {
        const icon = this.querySelector('.material-icons, i, svg');

        if (icon) {
            gsap.to(icon, {
                rotation: 360,
                scale: 0.8,
                duration: 0.5,
                ease: 'back.out(1.7)',
                onComplete: () => {
                    gsap.to(icon, {
                        scale: 1,
                        duration: 0.2
                    });
                }
            });
        }

        // Smooth page color transition
        gsap.to('body', {
            duration: 0.3,
            ease: 'power2.inOut'
        });
    });
}

// ===========================================
// STICKY NAVBAR ANIMATION
// ===========================================

function animateStickyNavbar() {
    const navbar = document.querySelector('.navbar, .topbar, header');
    if (!navbar) return;

    let lastScroll = 0;

    ScrollTrigger.create({
        start: 'top -100',
        end: 'max',
        onUpdate: (self) => {
            const currentScroll = self.scroll();

            if (currentScroll > lastScroll && currentScroll > 100) {
                // Scrolling down - shrink navbar
                gsap.to(navbar, {
                    y: -10,
                    scale: 0.98,
                    duration: 0.3,
                    ease: 'power2.out'
                });
            } else if (currentScroll < lastScroll) {
                // Scrolling up - expand navbar
                gsap.to(navbar, {
                    y: 0,
                    scale: 1,
                    duration: 0.4,
                    ease: 'back.out(1.4)'
                });
            }

            lastScroll = currentScroll;
        }
    });
}

// ===========================================
// UNIVERSAL HOVER STATES
// ===========================================

function enhanceHoverStates() {
    // Links: underline sweep
    const contentLinks = getAll('.docs-content a:not(.btn), .main-content a:not(.btn)');
    contentLinks.forEach(link => {
        link.addEventListener('mouseenter', function() {
            gsap.to(this, {
                x: 2,
                duration: 0.2,
                ease: 'power1.out'
            });
        });

        link.addEventListener('mouseleave', function() {
            gsap.to(this, {
                x: 0,
                duration: 0.2,
                ease: 'power1.in'
            });
        });
    });

    // Cards: lift effect
    const hoverCards = getAll('.card');
    hoverCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            gsap.to(this, {
                y: -5,
                boxShadow: '0 10px 30px rgba(0,0,0,0.15)',
                duration: 0.3,
                ease: 'power2.out'
            });
        });

        card.addEventListener('mouseleave', function() {
            gsap.to(this, {
                y: 0,
                boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
                duration: 0.3,
                ease: 'power2.in'
            });
        });
    });

    // Icons: rotate on hover
    const icons = getAll('.material-icons:not(.no-animate)');
    icons.forEach(icon => {
        icon.parentElement.addEventListener('mouseenter', function() {
            gsap.to(icon, {
                rotation: 15,
                scale: 1.1,
                duration: 0.3,
                ease: 'back.out(2)'
            });
        });

        icon.parentElement.addEventListener('mouseleave', function() {
            gsap.to(icon, {
                rotation: 0,
                scale: 1,
                duration: 0.3,
                ease: 'back.out(2)'
            });
        });
    });
}

// ===========================================
// INITIALIZATION
// ===========================================

function initGSAPAnimations() {
    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', runAnimations);
    } else {
        runAnimations();
    }
}

function runAnimations() {
    // Landing page animations
    animateHero();
    animateFeatureGrid();
    animateLandingSections();
    animateImageCompare();
    animateLandingHeader();

    // Documentation animations
    animateSidebar();
    animateTOC();
    animateBreadcrumbs();
    animateContent();
    animateCards();
    animateButtons();
    animateDocNav();
    animateTabs();
    animateBackToTop();
    animateDarkModeToggle();
    animateStickyNavbar();
    enhanceHoverStates();

    console.log('🎨 GSAP Animations initialized - PyMOCD Documentation');
}

// Auto-initialize
initGSAPAnimations();

// Export for manual initialization if needed
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { initGSAPAnimations };
}
