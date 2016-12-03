var socket = io.connect('http://' + document.domain + ':' + location.port);
socket.on('connect', function() {
    socket.emit('retrieve summary', {url: url});
    socket.on('message', function(msg) {
        var data = JSON.parse(msg);
        var summary = data.summary;
        console.log(data);

        updateProgress(data.progress);

        if (summary) {
            updateRating(summary.rating);
            updateTotalReviews(summary.review_count);
            updateSummaryText(summary.text);
            updateWords('negative-words', summary.negative_words);
            updateWords('positive-words', summary.positive_words);
        }
    });
});

function updateProgress(progress) {
    $('#summary-progress').css('width', progress + '%')
}

function updateRating(rating) {
    $('#summary-rating').text(rating);
}

function updateTotalReviews(totalReviews) {
    $('#summary-total-reviews').text(totalReviews);
}

function updateSummaryText(text) {
    $('#summary-text').text(text);
}

function updateWords(id, words) {
    var wordList = $('#' + id);
    wordList.empty();
    words.forEach(function(word) {
        var row = createWordRow(word.word, word.median_positivity);
        wordList.append(row);
    });
}

function createWordRow(word, percentPositive) {
    return'<p><span class="alert badge"><i class="fi-x"></i></span><span class="word">' + word + '</span><span class="stat float-right">' + percentPositive + '%</span></p>'
}
