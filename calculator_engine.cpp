// ============================================================================
// Advanced Calculator Engine for WebAssembly
// File: calculator_engine.cpp
// Compile with Emscripten:
//   emcc -std=c++17 -O3 calculator_engine.cpp -o calculator_engine.js \
//        -s EXPORTED_FUNCTIONS='["_malloc", "_free", "_evaluate", "_get_steps", "_graph"]' \
//        -s EXPORTED_RUNTIME_METHODS='["ccall", "cwrap"]' \
//        --bind
// ============================================================================

#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>

// ----------------------------------------------------------------------------
// Forward declarations
// ----------------------------------------------------------------------------
class Expression;
using ExprPtr = std::shared_ptr<Expression>;
using Complex = std::complex<double>;

// ----------------------------------------------------------------------------
// Environment: stores variables and functions
// ----------------------------------------------------------------------------
class Environment {
public:
    std::map<std::string, Complex> variables;
    std::map<std::string, std::function<Complex(const std::vector<Complex>&)>> functions;

    Environment() {
        // Predefined constants
        variables["pi"] = Complex(M_PI, 0);
        variables["e"]  = Complex(M_E, 0);
        variables["i"]  = Complex(0, 1);

        // Basic functions (real-valued, but accept complex)
        functions["sin"]   = [](const std::vector<Complex>& args) { return std::sin(args[0]); };
        functions["cos"]   = [](const std::vector<Complex>& args) { return std::cos(args[0]); };
        functions["tan"]   = [](const std::vector<Complex>& args) { return std::tan(args[0]); };
        functions["asin"]  = [](const std::vector<Complex>& args) { return std::asin(args[0]); };
        functions["acos"]  = [](const std::vector<Complex>& args) { return std::acos(args[0]); };
        functions["atan"]  = [](const std::vector<Complex>& args) { return std::atan(args[0]); };
        functions["sinh"]  = [](const std::vector<Complex>& args) { return std::sinh(args[0]); };
        functions["cosh"]  = [](const std::vector<Complex>& args) { return std::cosh(args[0]); };
        functions["tanh"]  = [](const std::vector<Complex>& args) { return std::tanh(args[0]); };
        functions["exp"]   = [](const std::vector<Complex>& args) { return std::exp(args[0]); };
        functions["log"]   = [](const std::vector<Complex>& args) { return std::log(args[0]); };   // natural log
        functions["log10"] = [](const std::vector<Complex>& args) { return std::log10(args[0]); };
        functions["sqrt"]  = [](const std::vector<Complex>& args) { return std::sqrt(args[0]); };
        functions["cbrt"]  = [](const std::vector<Complex>& args) { return std::pow(args[0], 1.0/3.0); };
        functions["abs"]   = [](const std::vector<Complex>& args) { return std::abs(args[0]); };
        functions["floor"] = [](const std::vector<Complex>& args) { return std::floor(args[0].real()); };
        functions["ceil"]  = [](const std::vector<Complex>& args) { return std::ceil(args[0].real()); };
        functions["round"] = [](const std::vector<Complex>& args) { return std::round(args[0].real()); };
        // Add more as needed...
    }

    Complex getVariable(const std::string& name) const {
        auto it = variables.find(name);
        if (it != variables.end()) return it->second;
        throw std::runtime_error("Unknown variable: " + name);
    }

    void setVariable(const std::string& name, Complex value) {
        variables[name] = value;
    }

    std::function<Complex(const std::vector<Complex>&)> getFunction(const std::string& name) const {
        auto it = functions.find(name);
        if (it != functions.end()) return it->second;
        throw std::runtime_error("Unknown function: " + name);
    }
};

// ----------------------------------------------------------------------------
// Expression base class
// ----------------------------------------------------------------------------
class Expression {
public:
    virtual ~Expression() = default;
    virtual Complex evaluate(const Environment& env) const = 0;
    virtual std::string toString() const = 0;
    // For step-by-step: return a simplified version (if possible) and a description
    virtual std::pair<ExprPtr, std::string> simplify() const { return {nullptr, ""}; }
};

// ----------------------------------------------------------------------------
// Concrete expression types
// ----------------------------------------------------------------------------

class NumberExpr : public Expression {
    Complex value;
public:
    NumberExpr(Complex v) : value(v) {}
    Complex evaluate(const Environment&) const override { return value; }
    std::string toString() const override {
        std::ostringstream oss;
        if (value.imag() == 0) oss << value.real();
        else oss << "(" << value.real() << "+" << value.imag() << "i)";
        return oss.str();
    }
};

class VariableExpr : public Expression {
    std::string name;
public:
    VariableExpr(const std::string& n) : name(n) {}
    Complex evaluate(const Environment& env) const override {
        return env.getVariable(name);
    }
    std::string toString() const override { return name; }
};

class BinaryOpExpr : public Expression {
public:
    enum Op { ADD, SUB, MUL, DIV, POW, LT, GT, LE, GE, EQ, NE };
private:
    Op op;
    ExprPtr left, right;
public:
    BinaryOpExpr(Op o, ExprPtr l, ExprPtr r) : op(o), left(l), right(r) {}

    Complex evaluate(const Environment& env) const override {
        Complex l = left->evaluate(env);
        Complex r = right->evaluate(env);
        switch (op) {
            case ADD: return l + r;
            case SUB: return l - r;
            case MUL: return l * r;
            case DIV: return l / r;
            case POW: return std::pow(l, r);
            // Comparison operators: return 1.0 if true, 0.0 if false (real part only)
            case LT: return (std::abs(l - r) < 1e-12 ? false : (l.real() < r.real())) ? 1.0 : 0.0;
            case GT: return (std::abs(l - r) < 1e-12 ? false : (l.real() > r.real())) ? 1.0 : 0.0;
            case LE: return (l.real() <= r.real()) ? 1.0 : 0.0;
            case GE: return (l.real() >= r.real()) ? 1.0 : 0.0;
            case EQ: return (std::abs(l - r) < 1e-12) ? 1.0 : 0.0;
            case NE: return (std::abs(l - r) >= 1e-12) ? 1.0 : 0.0;
            default: throw std::runtime_error("Unknown binary operator");
        }
    }

    std::string opString() const {
        switch (op) {
            case ADD: return "+"; case SUB: return "-"; case MUL: return "*";
            case DIV: return "/"; case POW: return "^";
            case LT: return "<"; case GT: return ">"; case LE: return "<=";
            case GE: return ">="; case EQ: return "=="; case NE: return "!=";
            default: return "?";
        }
    }

    std::string toString() const override {
        return "(" + left->toString() + " " + opString() + " " + right->toString() + ")";
    }

    std::pair<ExprPtr, std::string> simplify() const override {
        // Try constant folding
        auto lnum = std::dynamic_pointer_cast<NumberExpr>(left);
        auto rnum = std::dynamic_pointer_cast<NumberExpr>(right);
        if (lnum && rnum) {
            Complex result;
            // Evaluate using the same logic as evaluate (but without env)
            switch (op) {
                case ADD: result = lnum->evaluate(Environment()) + rnum->evaluate(Environment()); break;
                case SUB: result = lnum->evaluate(Environment()) - rnum->evaluate(Environment()); break;
                case MUL: result = lnum->evaluate(Environment()) * rnum->evaluate(Environment()); break;
                case DIV: result = lnum->evaluate(Environment()) / rnum->evaluate(Environment()); break;
                case POW: result = std::pow(lnum->evaluate(Environment()), rnum->evaluate(Environment())); break;
                // Comparisons could also be folded
                default: break;
            }
            return {std::make_shared<NumberExpr>(result), "Constant folding: " + toString() + " = " + NumberExpr(result).toString()};
        }
        // Identity rules: x+0 = x, x*1 = x, etc.
        if (op == ADD && rnum && rnum->evaluate(Environment()) == Complex(0,0))
            return {left, "Identity: " + toString() + " = " + left->toString()};
        if (op == ADD && lnum && lnum->evaluate(Environment()) == Complex(0,0))
            return {right, "Identity: " + toString() + " = " + right->toString()};
        if (op == MUL && rnum && rnum->evaluate(Environment()) == Complex(1,0))
            return {left, "Identity: " + toString() + " = " + left->toString()};
        if (op == MUL && lnum && lnum->evaluate(Environment()) == Complex(1,0))
            return {right, "Identity: " + toString() + " = " + right->toString()};
        if (op == MUL && (rnum && rnum->evaluate(Environment()) == Complex(0,0) ||
                          lnum && lnum->evaluate(Environment()) == Complex(0,0)))
            return {std::make_shared<NumberExpr>(0.0), "Multiplication by zero: " + toString() + " = 0"};

        return {nullptr, ""};
    }
};

class UnaryOpExpr : public Expression {
    std::string funcName;
    ExprPtr arg;
public:
    UnaryOpExpr(const std::string& name, ExprPtr a) : funcName(name), arg(a) {}

    Complex evaluate(const Environment& env) const override {
        auto f = env.getFunction(funcName);
        return f({arg->evaluate(env)});
    }

    std::string toString() const override {
        return funcName + "(" + arg->toString() + ")";
    }

    std::pair<ExprPtr, std::string> simplify() const override {
        auto argNum = std::dynamic_pointer_cast<NumberExpr>(arg);
        if (argNum) {
            Environment env;
            Complex result = evaluate(env);
            return {std::make_shared<NumberExpr>(result), "Evaluate function: " + toString() + " = " + NumberExpr(result).toString()};
        }
        return {nullptr, ""};
    }
};

// ----------------------------------------------------------------------------
// Parser (Recursive Descent)
// ----------------------------------------------------------------------------
class Parser {
    std::string input;
    size_t pos;
    std::string currentToken;
    Environment& env;  // only for function lookup

    void skipWhitespace() {
        while (pos < input.size() && isspace(input[pos])) pos++;
    }

    std::string getNextToken() {
        skipWhitespace();
        if (pos >= input.size()) return "";

        // Numbers (including complex? we parse as double, later we can add i suffix)
        if (isdigit(input[pos]) || input[pos] == '.') {
            size_t start = pos;
            while (pos < input.size() && (isdigit(input[pos]) || input[pos] == '.')) pos++;
            // Check for scientific notation
            if (pos < input.size() && (input[pos] == 'e' || input[pos] == 'E')) {
                pos++;
                if (pos < input.size() && (input[pos] == '+' || input[pos] == '-')) pos++;
                while (pos < input.size() && isdigit(input[pos])) pos++;
            }
            return input.substr(start, pos - start);
        }

        // Identifiers (variables or functions)
        if (isalpha(input[pos]) || input[pos] == '_') {
            size_t start = pos;
            while (pos < input.size() && (isalnum(input[pos]) || input[pos] == '_')) pos++;
            return input.substr(start, pos - start);
        }

        // Operators and punctuation
        std::string ops = "+-*/^()=<>!&|,";
        if (ops.find(input[pos]) != std::string::npos) {
            // Handle two-character operators: <=, >=, ==, !=, &&, ||
            if (pos + 1 < input.size()) {
                std::string two = input.substr(pos, 2);
                if (two == "<=" || two == ">=" || two == "==" || two == "!=" || two == "&&" || two == "||") {
                    pos += 2;
                    return two;
                }
            }
            return std::string(1, input[pos++]);
        }

        throw std::runtime_error("Unexpected character: " + std::string(1, input[pos]));
    }

    void nextToken() {
        currentToken = getNextToken();
    }

    bool match(const std::string& tok) {
        if (currentToken == tok) {
            nextToken();
            return true;
        }
        return false;
    }

    // Grammar:
    // expression = comparison
    // comparison = additive ( ('<' | '>' | '<=' | '>=' | '==' | '!=') additive )*
    // additive = multiplicative ( ('+' | '-') multiplicative )*
    // multiplicative = power ( ('*' | '/') power )*
    // power = unary ( '^' unary )*
    // unary = ( '+' | '-' | function-call )? primary
    // primary = number | variable | '(' expression ')'

    ExprPtr parseExpression() {
        return parseComparison();
    }

    ExprPtr parseComparison() {
        ExprPtr left = parseAdditive();
        while (currentToken == "<" || currentToken == ">" || currentToken == "<=" ||
               currentToken == ">=" || currentToken == "==" || currentToken == "!=") {
            std::string op = currentToken;
            nextToken();
            ExprPtr right = parseAdditive();
            BinaryOpExpr::Op bop;
            if (op == "<") bop = BinaryOpExpr::LT;
            else if (op == ">") bop = BinaryOpExpr::GT;
            else if (op == "<=") bop = BinaryOpExpr::LE;
            else if (op == ">=") bop = BinaryOpExpr::GE;
            else if (op == "==") bop = BinaryOpExpr::EQ;
            else if (op == "!=") bop = BinaryOpExpr::NE;
            else throw std::runtime_error("Unknown comparison op");
            left = std::make_shared<BinaryOpExpr>(bop, left, right);
        }
        return left;
    }

    ExprPtr parseAdditive() {
        ExprPtr left = parseMultiplicative();
        while (currentToken == "+" || currentToken == "-") {
            std::string op = currentToken;
            nextToken();
            ExprPtr right = parseMultiplicative();
            BinaryOpExpr::Op bop = (op == "+") ? BinaryOpExpr::ADD : BinaryOpExpr::SUB;
            left = std::make_shared<BinaryOpExpr>(bop, left, right);
        }
        return left;
    }

    ExprPtr parseMultiplicative() {
        ExprPtr left = parsePower();
        while (currentToken == "*" || currentToken == "/") {
            std::string op = currentToken;
            nextToken();
            ExprPtr right = parsePower();
            BinaryOpExpr::Op bop = (op == "*") ? BinaryOpExpr::MUL : BinaryOpExpr::DIV;
            left = std::make_shared<BinaryOpExpr>(bop, left, right);
        }
        return left;
    }

    ExprPtr parsePower() {
        ExprPtr left = parseUnary();
        while (currentToken == "^") {
            nextToken();
            ExprPtr right = parseUnary();
            left = std::make_shared<BinaryOpExpr>(BinaryOpExpr::POW, left, right);
        }
        return left;
    }

    ExprPtr parseUnary() {
        if (currentToken == "+" || currentToken == "-") {
            std::string op = currentToken;
            nextToken();
            ExprPtr sub = parseUnary();
            if (op == "-") {
                // Unary minus: treat as 0 - sub
                auto zero = std::make_shared<NumberExpr>(0.0);
                return std::make_shared<BinaryOpExpr>(BinaryOpExpr::SUB, zero, sub);
            }
            return sub; // unary plus does nothing
        }
        // Function call?
        if (isalpha(currentToken[0])) {
            std::string name = currentToken;
            // Look ahead: if next token is '(', it's a function call, else variable
            size_t savedPos = pos;
            std::string savedToken = currentToken;
            nextToken(); // peek after identifier
            if (currentToken == "(") {
                // It's a function
                std::string funcName = name;
                nextToken(); // consume '('
                std::vector<ExprPtr> args;
                if (currentToken != ")") {
                    args.push_back(parseExpression());
                    while (match(",")) {
                        args.push_back(parseExpression());
                    }
                }
                if (!match(")")) throw std::runtime_error("Expected ')' after function arguments");
                if (args.size() != 1) {
                    // For simplicity we only handle unary functions now; can be extended
                    throw std::runtime_error("Only unary functions supported in this demo");
                }
                return std::make_shared<UnaryOpExpr>(funcName, args[0]);
            } else {
                // It's a variable
                pos = savedPos;
                currentToken = savedToken;
                nextToken(); // consume variable token
                return std::make_shared<VariableExpr>(name);
            }
        }
        return parsePrimary();
    }

    ExprPtr parsePrimary() {
        if (currentToken == "(") {
            nextToken();
            ExprPtr expr = parseExpression();
            if (!match(")")) throw std::runtime_error("Expected ')'");
            return expr;
        }
        // Number
        if (isdigit(currentToken[0]) || currentToken[0] == '.') {
            double val = std::stod(currentToken);
            nextToken();
            return std::make_shared<NumberExpr>(val);
        }
        throw std::runtime_error("Unexpected token: " + currentToken);
    }

public:
    Parser(const std::string& expr, Environment& e) : input(expr), pos(0), env(e) {
        nextToken();
    }

    ExprPtr parse() {
        ExprPtr expr = parseExpression();
        if (!currentToken.empty()) throw std::runtime_error("Unexpected trailing characters: " + currentToken);
        return expr;
    }
};

// ----------------------------------------------------------------------------
// Calculator Engine: main interface
// ----------------------------------------------------------------------------
class CalculatorEngine {
    Environment env;
    std::vector<std::string> steps;  // stores step-by-step explanations

public:
    CalculatorEngine() {
        // default environment already set up
    }

    // Set a variable (exposed to JS)
    void setVariable(const std::string& name, double real, double imag = 0.0) {
        env.setVariable(name, Complex(real, imag));
    }

    // Evaluate an expression and return the result as a string (including steps)
    std::string evaluate(const std::string& exprStr, bool recordSteps = true) {
        steps.clear();
        try {
            Parser parser(exprStr, env);
            ExprPtr expr = parser.parse();

            if (recordSteps) {
                // Perform simplification steps until no more simplifications
                ExprPtr current = expr;
                while (true) {
                    auto [simplified, desc] = current->simplify();
                    if (!simplified) break;
                    steps.push_back(desc);
                    current = simplified;
                }
                // Final evaluation
                Complex result = current->evaluate(env);
                steps.push_back("Result = " + numberToString(result));
                return steps.back();
            } else {
                Complex result = expr->evaluate(env);
                return numberToString(result);
            }
        } catch (const std::exception& e) {
            return std::string("Error: ") + e.what();
        }
    }

    // Get the recorded steps as a single string (newline separated)
    std::string getSteps() const {
        std::ostringstream oss;
        for (const auto& s : steps) oss << s << "\n";
        return oss.str();
    }

    // Graph a function of one variable (x) over [xmin, xmax] with n points
    // Returns a string of JSON-like array: [[x1,y1], [x2,y2], ...]
    std::string graph(const std::string& exprStr, double xmin, double xmax, int n = 100) {
        std::ostringstream oss;
        oss << "[";
        try {
            // Temporarily store original x value
            Complex originalX = env.getVariable("x");
            double step = (xmax - xmin) / (n - 1);
            for (int i = 0; i < n; ++i) {
                double x = xmin + i * step;
                env.setVariable("x", x);
                Parser parser(exprStr, env);
                ExprPtr expr = parser.parse();
                Complex y = expr->evaluate(env);
                if (i > 0) oss << ",";
                oss << "[" << x << "," << y.real() << "]"; // only real part for graphing
            }
            // Restore x
            env.setVariable("x", originalX);
        } catch (const std::exception& e) {
            oss << "{\"error\":\"" << e.what() << "\"}";
        }
        oss << "]";
        return oss.str();
    }

private:
    static std::string numberToString(Complex z) {
        std::ostringstream oss;
        double r = z.real();
        double i = z.imag();
        if (std::abs(i) < 1e-12) {
            oss << r;
        } else {
            oss << "(" << r << (i >= 0 ? "+" : "") << i << "i)";
        }
        return oss.str();
    }
};

// ----------------------------------------------------------------------------
// C API for WebAssembly (using simple C strings)
// ----------------------------------------------------------------------------
extern "C" {

// Global engine instance (for simplicity; in production you might want multiple)
CalculatorEngine* engine = nullptr;

// Initialize engine (must be called once)
void init_engine() {
    if (!engine) engine = new CalculatorEngine();
}

// Free engine
void free_engine() {
    delete engine;
    engine = nullptr;
}

// Evaluate expression and return result as C string (caller must free with free_string)
const char* evaluate(const char* expr, int recordSteps) {
    if (!engine) init_engine();
    std::string result = engine->evaluate(expr, recordSteps != 0);
    char* cstr = (char*)malloc(result.size() + 1);
    strcpy(cstr, result.c_str());
    return cstr;
}

// Get last steps as C string
const char* get_steps() {
    if (!engine) return strdup("");
    std::string steps = engine->getSteps();
    char* cstr = (char*)malloc(steps.size() + 1);
    strcpy(cstr, steps.c_str());
    return cstr;
}

// Set variable (real and imag)
void set_variable(const char* name, double real, double imag) {
    if (!engine) init_engine();
    engine->setVariable(name, real, imag);
}

// Graph function: returns JSON string
const char* graph(const char* expr, double xmin, double xmax, int n) {
    if (!engine) init_engine();
    std::string data = engine->graph(expr, xmin, xmax, n);
    char* cstr = (char*)malloc(data.size() + 1);
    strcpy(cstr, data.c_str());
    return cstr;
}

// Free a string returned by the above functions
void free_string(char* str) {
    free(str);
}

} // extern "C"

// ----------------------------------------------------------------------------
// Optional: test main (if compiled natively)
// ----------------------------------------------------------------------------
#ifdef TEST_MAIN
int main() {
    CalculatorEngine calc;
    std::string expr = "sin(pi/2) + 2*3";
    std::cout << "Evaluating: " << expr << std::endl;
    std::string result = calc.evaluate(expr, true);
    std::cout << "Steps:\n" << calc.getSteps() << std::endl;
    std::cout << "Result: " << result << std::endl;

    // Test graphing
    std::string graphData = calc.graph("x^2", -2.0, 2.0, 5);
    std::cout << "Graph data: " << graphData << std::endl;
    return 0;
}
#endif
