// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//

// RAG example for fruit recipes of using alpaca-core's llama inference

// llama
#include <ac/llama/Init.hpp>
#include <ac/llama/Model.hpp>
#include <ac/llama/Instance.hpp>
#include <ac/llama/InstanceEmbedding.hpp>
#include <ac/llama/Session.hpp>
#include <ac/llama/ResourceCache.hpp>
#include <ac/llama/ChatFormat.hpp>

// hnswlib
#include <hnswlib/hnswlib.h>

// logging
#include <ac/jalog/Instance.hpp>
#include <ac/jalog/sinks/ColorSink.hpp>

// model source directory
#include "ac-test-data-llama-dir.h"

#include <iostream>
#include <string>



std::vector<std::pair<std::string, std::string>> g_recipes = {
    {"Apple Banana", "Apple Banana Smoothie: Blend 1 apple, 1 banana, 1 cup milk, and 1 tbsp honey until smooth."},
    {"Orange Strawberry", "Orange Strawberry Salad: Toss 1 cup sliced strawberries with 1/2 cup orange segments, 1 tbsp honey, and fresh mint."},
    {"Blueberry Raspberry", "Blueberry Raspberry Muffins: Mix 1 1/2 cups flour, 3/4 cup sugar, 1 tsp baking powder, 1 egg, 1/2 cup milk, and 1/4 cup oil. Fold in 1/2 cup blueberries and 1/2 cup raspberries, then bake at 375°F (190°C) for 20 minutes."},
    {"Blackberry Mango", "Mango Blackberry Parfait: Layer 1 cup yogurt, 1/2 cup diced mango, 1/2 cup blackberries, and 1/4 cup granola."},
    {"Peach Plum", "Peach Plum Crumble: Mix 2 cups sliced peaches and 2 cups sliced plums with 1/4 cup sugar. Top with a crumble of 1/2 cup oats, 1/4 cup flour, 1/4 cup butter, and 1/4 cup sugar. Bake at 375°F (190°C) for 30 minutes."},
    {"Cherry Grapes", "Cherry Grape Clafoutis: Blend 1 cup milk, 3 eggs, 1/2 cup flour, 1/2 cup sugar. Pour over 1 cup pitted cherries and 1 cup halved grapes in a baking dish, then bake at 375°F (190°C) for 40 minutes."},
    {"Watermelon Kiwi", "Watermelon Kiwi Cooler: Blend 2 cups watermelon chunks with 1 sliced kiwi and 1 tbsp lime juice. Serve chilled."},
    {"Apple Orange", "Apple Orange Salad: Toss 1 diced apple, 1/2 cup orange segments, 1/4 cup walnuts, and 2 tbsp yogurt dressing."},
    {"Banana Blueberry", "Banana Blueberry Pancakes: Mash 2 ripe bananas, mix with 1 cup flour, 1 tsp baking powder, 1 egg, 1/2 cup milk, and 1/2 cup blueberries. Cook pancakes on medium heat until golden brown."},
    {"Strawberry Raspberry", "Berry Shortcake: Slice 1 cup strawberries and 1 cup raspberries, mix with 2 tbsp sugar. Serve over biscuits with whipped cream."},
    {"Mango Peach", "Mango Peach Salsa: Dice 1 mango and 1 peach, mix with 1/4 cup red onion, 1 tbsp lime juice, and 1 tbsp chopped cilantro."},
    {"Plum Blackberry", "Plum Blackberry Tart: Layer sliced plums and blackberries over a pie crust. Mix 1/4 cup sugar and 1 tbsp flour, sprinkle over fruit, bake at 375°F (190°C) for 30 minutes."},
    {"Cherry Watermelon", "Cherry Watermelon Salad: Mix 1 cup pitted cherries, 2 cups watermelon chunks, 1 tbsp balsamic glaze, and fresh mint."},
    {"Grapes Kiwi", "Grape Kiwi Smoothie: Blend 1 cup grapes, 2 sliced kiwis, 1/2 cup yogurt, and 1/2 cup ice."},
    {"Apple Raspberry", "Apple Raspberry Crumble: Mix 2 cups sliced apples and 1 cup raspberries with 1/4 cup sugar. Top with a crumble of 1/2 cup oats, 1/4 cup flour, 1/4 cup butter, and 1/4 cup sugar. Bake at 375°F (190°C) for 30 minutes."},
    {"Orange Blackberry", "Citrus Blackberry Glaze: Simmer 1/2 cup orange juice, 1/2 cup blackberries, 2 tbsp honey, and 1 tbsp balsamic vinegar. Drizzle over grilled chicken."},
    {"Blueberry Cherry", "Blueberry Cherry Cobbler: Mix 1 cup blueberries and 1 cup cherries with 1/2 cup sugar and 1 tbsp lemon juice. Top with a batter of 1 cup flour, 1/2 cup sugar, 1 tsp baking powder, 1/2 cup milk, and bake at 375°F (190°C) for 30 minutes."},
    {"Peach Watermelon", "Peach Watermelon Sorbet: Blend 2 cups watermelon, 1 cup peaches, and 1/4 cup honey. Freeze until firm."},
    {"Strawberry Mango", "Strawberry Mango Lassi: Blend 1/2 cup strawberries, 1/2 cup mango, 1 cup yogurt, 1 tbsp honey, and 1/2 cup ice."},
    {"Plum Grapes", "Plum Grape Chutney: Simmer 1 cup plums and 1 cup grapes with 1/4 cup sugar, 1 tbsp vinegar, and 1/2 tsp cinnamon until thickened."},
    {"Raspberry Kiwi", "Raspberry Kiwi Yogurt Bowl: Layer 1 cup yogurt, 1/2 cup raspberries, 1 sliced kiwi, and 1/4 cup granola."},
    {"Banana Peach", "Banana Peach Oatmeal: Cook 1/2 cup oats with 1 mashed banana and 1/2 cup diced peaches. Top with nuts and honey."},
    {"Apple Blackberry", "Apple Blackberry Pie: Mix 2 cups sliced apples, 1 cup blackberries, 3/4 cup sugar, 1 tsp cinnamon, and 1/4 cup flour. Fill a pie crust, top with another crust, and bake at 375°F (190°C) for 50 minutes."},
    {"Orange Blueberry", "Orange Blueberry Muffins: Mix 1 1/2 cups flour, 3/4 cup sugar, 1 tsp baking powder, 1 egg, 1/2 cup milk, 1 tbsp orange zest, and 1/4 cup oil. Fold in 1 cup blueberries and bake at 375°F (190°C) for 20 minutes."},
    {"Strawberry Cherry", "Strawberry Cherry Jam: Cook 2 cups strawberries, 1 cup cherries, and 1/2 cup sugar until thick. Store in jars."},
    {"Mango Kiwi", "Mango Kiwi Popsicles: Blend 1 cup mango and 1 cup kiwi with 1 tbsp honey. Pour into molds and freeze."},
    {"Blackberry Grapes", "Blackberry Grape Smoothie: Blend 1 cup blackberries, 1 cup grapes, 1/2 cup yogurt, and 1/2 cup milk."},
    {"Banana Watermelon", "Banana Watermelon Ice Cream: Blend 2 frozen bananas with 1 cup watermelon. Freeze until firm."},
    {"Apple Orange", "Apple Orange Juice: Blend 2 apples and 1 peeled orange with 1/2 cup water. Strain and serve."}
};

ac::llama::Instance* g_chatInstance;
ac::llama::InstanceEmbedding* g_embeddingInstance;
std::vector<ac::llama::ChatMsg> g_messages;

template<typename T>
class VectorDatabase {
public:
    VectorDatabase(int dim, int max_elements)
        : m_dim(dim)
        , m_space(dim)
        , m_index(&m_space, max_elements, dim, 200)
    {}

    void addEntry(const std::vector<float>& embedding, int idx, T content) {
        assert(embedding.size() == (size_t)m_dim);
        m_index.addPoint(embedding.data(), idx);
        m_documents[idx] = content;
    }

    struct Result {
        float dist;
        int idx;
        T content;

        bool operator<(const Result& other) const {
            return dist < other.dist;
        }

        bool operator>(const Result& other) const {
            return dist > other.dist;
        }
    };
    using SearchResults = std::priority_queue<Result, std::vector<Result>, std::greater<Result>>;
    SearchResults searchKnn(const std::vector<float>& query, int k) {
        auto results = m_index.searchKnn(query.data(), k);

        SearchResults searchResults;
        while (!results.empty()) {
            auto res = results.top();

            auto doc = m_documents[res.second];
            searchResults.emplace(Result{
                .dist = res.first,
                .idx = (int)res.second,
                .content = doc
            });

            results.pop();
        }

        return searchResults;
    }

    void saveIndex(const std::string& location) {
        std::string db = location + "db.bin";
        m_index.saveIndex(db);

        std::string documents = location + "documents.bin";
        std::ofstream out(documents, std::ios::binary);
        for (const auto& [idx, content] : m_documents) {
            out.write(reinterpret_cast<const char*>(&idx), sizeof(idx));
            content.write(out);
        }
        out.close();
    }

    void loadIndex(const std::string& location) {
        const std::string db = location + "db.bin";
        m_index.loadIndex(db, &m_space);

        std::string documents = location + "documents.bin";
        std::ifstream in(documents, std::ios::binary);
        while (in) {
            int idx;
            T content;
            in.read(reinterpret_cast<char*>(&idx), sizeof(idx));
            if (in.eof()) {
                break;
            }
            content.read(in);
            m_documents[idx] = content;
        }
        in.close();
    }

private:
    int m_dim;
    std::unordered_map<int, T> m_documents;

    hnswlib::L2Space m_space;
    hnswlib::HierarchicalNSW<float> m_index;

};

// This is a simple object for storing a document
// You can modify it to store custom information if it's needed
// Note: Do not forget to implement the write and read functions for your custom object
struct Document {
    std::string content;
    void write(std::ostream& out) const {
        size_t strSize = content.size();

        out.write(reinterpret_cast<const char*>(&strSize), sizeof(strSize));
        out.write(content.data(), strSize * sizeof(char));
    }

    void read(std::istream& in) {
        size_t strSize = 0;
        in.read(reinterpret_cast<char*>(&strSize), sizeof(strSize));
        content.resize(strSize);
        in.read(content.data(), strSize * sizeof(char));
    }
};

std::string retrieveKnowledge(const std::string& query, VectorDatabase<Document>& vdb) {
    // Search for the most relevant recipes
    // in the vector database
    int k = 3;  // Retrieve top-3 results
    std::vector<float> seachEmbedding = g_embeddingInstance->getEmbeddingVector(g_embeddingInstance->model().vocab().tokenize(query, true, true));
    auto results = vdb.searchKnn(seachEmbedding, k);

    std::string knowledgeContent = "";

    int cnt = 1;
    std::cout << "============= database context =============\n";
    while (!results.empty()) {
        auto res = results.top();
            std::cout << "\t" << cnt << "."
                    << " Distance: " << res.dist
                    << " ID:"         << res.idx
                    << " content: "   << res.content.content
                    << std::endl;
                    results.pop();
        knowledgeContent += std::to_string(cnt++) + ": " + res.content.content + "\n";
    }
    std::cout << "=============================================\n";

    return knowledgeContent;
}

std::string generateResponse(ac::llama::Session& session, const std::string& prompt, VectorDatabase<Document>& vdb, int maxTokens = 512) {
    ac::llama::ChatFormat chatFormat("llama3");
    ac::llama::ChatMsg msg{.role = "user", .text = prompt};

    // 1. Fill the context with the relevant recipes
    const std::string systemPrompt = "You are a recipe assistant. Given the following relevant recipes, select the most relevant one or paraphrase it:\n";
    const std::string knowledge = retrieveKnowledge(prompt, vdb);
    session.pushPrompt(g_chatInstance->model().vocab().tokenize(systemPrompt + knowledge, false, false));

    // 2. Add the user prompt to the context
    auto formatted = chatFormat.formatMsg(msg, g_messages, true);
    g_messages.emplace_back(msg);
    // Note: To format the full chat and push it into the context uncomment the following line
    // formatted = chatFormat.formatChat(g_messages, true);
    session.pushPrompt(g_chatInstance->model().vocab().tokenize(formatted, false, false));

    // 3. Generate the response
    std::string response = "";
    for (int i = 0; i < maxTokens; ++i) {
        auto token = session.getToken();
        if (token == ac::llama::Token_Invalid) {
            // no more tokens
            break;
        }

        if (g_chatInstance->model().vocab().isEog(token)) {
            response += '\n';
            break;
        }

        response += g_chatInstance->model().vocab().tokenToString(token);
    }

    g_messages.emplace_back(ac::llama::ChatMsg{.role = "system", .text = response});

    return response;
}

void fillDatabase(VectorDatabase<Document>& vdb) {
    for (size_t i = 0; i < g_recipes.size(); i++) {
        std::vector<float> embedding = g_embeddingInstance->getEmbeddingVector(g_embeddingInstance->model().vocab().tokenize(g_recipes[i].first , true, true));
        vdb.addEntry(embedding, i, Document{.content = g_recipes[i].second});
    }
}

int main() try {
    ac::jalog::Instance jl;
    jl.setup().add<ac::jalog::sinks::ColorSink>();

    // initialize the library
    ac::llama::initLibrary();

    // Note: This model won't work for this example, but it's a placeholder.
    //       Download better model - llama3.2 8b for example
    std::string modelGguf = AC_TEST_DATA_LLAMA_DIR "/gpt2-117m-q6_k.gguf";
    std::string embeddingModelGguf = AC_TEST_DATA_LLAMA_DIR "/bge-small-en-v1.5-f16.gguf";

    ac::local::ResourceManager rm;
    ac::llama::ResourceCache cache(rm);

    // create objects for the inference
    auto model = cache.getModel({.gguf = modelGguf});
    ac::llama::Instance instance(*model, {});
    g_chatInstance = &instance;

    // create objects for the embedding
    auto mEmbedding = cache.getModel({.gguf = embeddingModelGguf});
    ac::llama::InstanceEmbedding iEmbedding(*mEmbedding, {});
    g_embeddingInstance = &iEmbedding;

    uint32_t dim = iEmbedding.embeddingDim();
    VectorDatabase<Document> vdb(dim, 100);

#if 1
    fillDatabase(vdb);
    vdb.saveIndex("");
#else
    vdb.loadIndex("");
#endif

    // start session
    auto& session = instance.startSession({});
    const char* initialPrompt = "You are the perfect AI assistant that analyzes the provided information and give concise answer to the user";
    session.setInitialPrompt(instance.model().vocab().tokenize(initialPrompt, true, true));

    std::string lastResponse = "initial";
    while(true) {
        std::string prompt = "";
        if (!lastResponse.empty()) {
            std::cout << "\nUser: ";
        }
        std::cin.clear();
        std::cin.sync();
        std::getline(std::cin, prompt);
        if (prompt.empty()) {
            lastResponse = "";
            continue;
        }
        if (prompt == "exit") {
            break;
        }
        lastResponse = generateResponse(session, prompt, vdb);
        std::cout << "\nAI: " << lastResponse << std::endl;
    }

    return 0;
}
catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
}
