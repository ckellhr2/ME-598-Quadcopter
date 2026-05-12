#using Random
using DataFrames
using CSV
using Serialization

#EDITME  # placeholder for physics engine import
# using PyBullet or equivalent Julia package

include("IdealSystem.jl")  # assumes run_simulation is defined here

const GOAL_POS = [1.3, 2.1, 1.4]
const N_TRIALS = 10
const START_BOUNDS = Dict(
    "x" => (-1.5, 1.5),
    "y" => (-1.5, 1.5),
    "z" => (0.8, 1.8),
)

function sample_initial_condition(rng)
    return [
        rand(rng, Uniform(START_BOUNDS["x"]...)),
        rand(rng, Uniform(START_BOUNDS["y"]...)),
        rand(rng, Uniform(START_BOUNDS["z"]...)),
    ]
end

function main()
    base_seed = 598
    rng = MersenneTwister(base_seed)
    script_dir = @__DIR__

    #physics_client = #EDITME  # connect to physics engine
    results = []

    try
        for trial_index in 1:N_TRIALS
            seed = rand(rng, 0:(2^31 - 1))
            start_pos = sample_initial_condition(rng)

            result = run_simulation(
                start_pos = start_pos,
                x_goal_pos = GOAL_POS,
                #connection_mode = #EDITME,
                keep_alive = false,
                log_interval = 0,
                verbose = false,
                reuse_existing_connection = true,
            )

            result["trial"] = trial_index # changed from trial_index - 1
            result["seed"] = seed
            result["status"] = "ok"
            push!(results, result)

            println("Completed $trial_index/$N_TRIALS trials... cost=$(result["trajectory_total_cost"])")
        end
    finally
        #if #EDITME  # isConnected
            #EDITME  # disconnect
        #end
    end

    df = DataFrame(results)
    sort!(df, :trial)

    csv_path = joinpath(script_dir, "random_initial_condition_costs.csv")
    pickle_path = joinpath(script_dir, "random_initial_condition_costs.pkl")
    summary_path = joinpath(script_dir, "random_initial_condition_costs_summary.txt")

    CSV.write(csv_path, df)
    serialize(pickle_path, df)

    summary_lines = [
        "Batch complete.",
        "Trials: $(nrow(df))",
        "Goal position: $(GOAL_POS)",
        @sprintf("Mean cost: %.6f", mean(df.trajectory_total_cost)),
        @sprintf("Std cost: %.6f", std(df.trajectory_total_cost)),
        @sprintf("Min cost: %.6f", minimum(df.trajectory_total_cost)),
        @sprintf("Max cost: %.6f", maximum(df.trajectory_total_cost)),
        @sprintf("Success rate: %.3f", mean(df.reached_goal)),
        "CSV saved to: $csv_path",
        "Pickle saved to: $pickle_path",
    ]

    open(summary_path, "w") do f
        for line in summary_lines
            println(f, line)
        end
    end

    println("\nBatch complete.")
    println("Trials: $(nrow(df))")
    println(@sprintf("Mean cost: %.6f", mean(df.trajectory_total_cost)))
    println(@sprintf("Std cost: %.6f", std(df.trajectory_total_cost)))
    println("CSV saved to: $csv_path")
    println("Pickle saved to: $pickle_path")
    println("Summary saved to: $summary_path")
end

main()
