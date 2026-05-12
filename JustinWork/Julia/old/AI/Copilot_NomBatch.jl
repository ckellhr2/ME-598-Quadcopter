using Random
using DataFrames
using CSV
using Serialization

#EDITME  # placeholder for physics engine import
# using IdealSystem: run_simulation   # adjust as needed
include("IdealSystem.jl")  # assuming your Julia version lives here

const GOAL_POS = [1.3, 2.1, 1.4]
const N_TRIALS = 100

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

    output_csv_path = joinpath(script_dir, "nominal_trajectory_positions.csv")
    output_pickle_path = joinpath(script_dir, "nominal_trajectory_positions.pkl")
    output_summary_path = joinpath(script_dir, "nominal_trajectory_positions_summary.txt")

    #physics_client = #EDITME   # p.connect(p.DIRECT)

    trajectory_rows = Vector{Dict{String,Any}}()

    try
        println("Starting nominal trajectory batch.")
        println("Trials to run: $N_TRIALS")
        println("Goal position: $GOAL_POS")

        for trial_index in 0:(N_TRIALS-1)
            seed = rand(rng, 0:(2^31 - 1))
            start_pos = sample_initial_condition(rng)

            result = run_simulation(
                start_pos = start_pos,
                x_goal_pos = GOAL_POS,
                connection_mode = #EDITME,   # p.DIRECT
                keep_alive = false,
                log_interval = 0,
                verbose = false,
                reuse_existing_connection = true,
            )

            for row in result["trajectory_rows"]
                push!(trajectory_rows, Dict(
                    "trial" => trial_index,
                    "seed" => seed,
                    "start_x" => start_pos[1],
                    "start_y" => start_pos[2],
                    "start_z" => start_pos[3],
                    "goal_x" => GOAL_POS[1],
                    "goal_y" => GOAL_POS[2],
                    "goal_z" => GOAL_POS[3],
                    "time_step" => row["time_step"],
                    "time_sec" => row["time_sec"],
                    "x" => row["x"],
                    "y" => row["y"],
                    "z" => row["z"],
                ))
            end

            println(
                "Completed trial $(trial_index + 1)/$N_TRIALS... " *
                "stored $(length(result["trajectory_rows"])) time steps, " *
                "reached_goal=$(result["reached_goal"])"
            )
        end

    finally
        #if #EDITME   # p.isConnected(physics_client)
            #EDITME   # p.disconnect(physics_client)
        #end
    end

    trajectory_df = DataFrame(trajectory_rows)
    CSV.write(output_csv_path, trajectory_df)
    serialize(output_pickle_path, trajectory_df)

    summary_lines = [
        "Nominal trajectory batch complete.",
        "Trials: $(length(unique(trajectory_df.trial)))",
        "Stored rows: $(nrow(trajectory_df))",
        "Goal position: $GOAL_POS",
        "Max time step: $(maximum(trajectory_df.time_step))",
        "CSV saved to: $output_csv_path",
        "Pickle saved to: $output_pickle_path",
    ]

    open(output_summary_path, "w") do f
        for line in summary_lines
            println(f, line)
        end
    end

    println("\nNominal trajectory batch complete.")
    println("Trials: $(length(unique(trajectory_df.trial)))")
    println("Stored rows: $(nrow(trajectory_df))")
    println("CSV saved to: $output_csv_path")
    println("Pickle saved to: $output_pickle_path")
    println("Summary saved to: $output_summary_path")
end

main()
