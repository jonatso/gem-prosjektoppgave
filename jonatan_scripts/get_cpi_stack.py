from get_single_stat import get_single_stat


def get_cpi_stack(input_file_path):
    # Get the number of compute units
    num_cus = 0
    while (
        get_single_stat(input_file_path, f"system.cpu1.CUs{num_cus}.")
        is not None
    ):
        num_cus += 1

    print(f"Number of CUs: {num_cus}")

    cpi_stacks = {}
    for cu_id in range(num_cus):
        cu_cpi_stack = {}
        for wf_id in range(10):
            exec_stage_cycles_with_instr = get_single_stat(
                input_file_path,
                f"system.cpu1.CUs{cu_id}.wavefronts{wf_id}.ExecStage.numCyclesWithInstrIssued",
            )
            exec_stage_cycles_no_instr = get_single_stat(
                input_file_path,
                f"system.cpu1.CUs{cu_id}.wavefronts{wf_id}.ExecStage.numCyclesWithNoIssue",
            )
            scoreboard_stage_stall_cycles = get_single_stat(
                input_file_path,
                f"system.cpu1.CUs{cu_id}.wavefronts{wf_id}.ScoreboardCheckStage.stallCycles",
            )
            schedule_stage_dispatch_list_cycles = get_single_stat(
                input_file_path,
                f"system.cpu1.CUs{cu_id}.wavefronts{wf_id}.ScheduleStage.schListToDispList",
            )
            schedule_stage_dispatch_list_stall_cycles = get_single_stat(
                input_file_path,
                f"system.cpu1.CUs{cu_id}.wavefronts{wf_id}.ScheduleStage.schListToDispListStalls",
            )
            schedule_stage_rf_access_stall_cycles = get_single_stat(
                input_file_path,
                f"system.cpu1.CUs{cu_id}.wavefronts{wf_id}.ScheduleStage.rfAccessStalls",
            )
            schedule_stage_lds_stall_cycles = get_single_stat(
                input_file_path,
                f"system.cpu1.CUs{cu_id}.wavefronts{wf_id}.ScheduleStage.ldsBusArbStalls",
            )
            schedule_stage_operand_stall_cycles = get_single_stat(
                input_file_path,
                f"system.cpu1.CUs{cu_id}.wavefronts{wf_id}.ScheduleStage.opdNrdyStalls",
            )
            schedule_stage_resource_stall_cycles = get_single_stat(
                input_file_path,
                f"system.cpu1.CUs{cu_id}.wavefronts{wf_id}.ScheduleStage.dispNrdyStalls",
            )
            wavefront_schedule_stage_cycles = get_single_stat(
                input_file_path,
                f"system.cpu1.CUs{cu_id}.wavefronts{wf_id}.Wavefront.schCycles",
            )
            wavefront_schedule_stage_stall_cycles = get_single_stat(
                input_file_path,
                f"system.cpu1.CUs{cu_id}.wavefronts{wf_id}.Wavefront.schStalls",
            )
            wavefront_schedule_stage_rf_access_stall_cycles = get_single_stat(
                input_file_path,
                f"system.cpu1.CUs{cu_id}.wavefronts{wf_id}.Wavefront.schRfAccessStalls",
            )
            wavefront_schedule_stage_resource_stall_cycles = get_single_stat(
                input_file_path,
                f"system.cpu1.CUs{cu_id}.wavefronts{wf_id}.Wavefront.schResourceStalls",
            )
            wavefront_schedule_stage_operand_stall_cycles = get_single_stat(
                input_file_path,
                f"system.cpu1.CUs{cu_id}.wavefronts{wf_id}.Wavefront.schOpdNrdyStalls",
            )
            wavefront_schedule_stage_lds_stall_cycles = get_single_stat(
                input_file_path,
                f"system.cpu1.CUs{cu_id}.wavefronts{wf_id}.Wavefront.schLdsArbStalls",
            )
            total_instructions = get_single_stat(
                input_file_path,
                f"system.cpu1.CUs{cu_id}.wavefronts{wf_id}.Wavefront.numInstrExecuted",
            )

            if (
                exec_stage_cycles_with_instr is not None
                and exec_stage_cycles_no_instr is not None
                and total_instructions is not None
            ):
                cpi = (
                    exec_stage_cycles_with_instr + exec_stage_cycles_no_instr
                ) / total_instructions
                cpi_stack = {
                    "Exec Stage": exec_stage_cycles_with_instr
                    / total_instructions,
                    "No Issue": exec_stage_cycles_no_instr
                    / total_instructions,
                    "SCB Stage": scoreboard_stage_stall_cycles
                    / total_instructions
                    if scoreboard_stage_stall_cycles is not None
                    else 0.0,
                    "SCH Stage": (
                        schedule_stage_dispatch_list_cycles
                        + schedule_stage_dispatch_list_stall_cycles
                    )
                    / total_instructions
                    if schedule_stage_dispatch_list_cycles is not None
                    and schedule_stage_dispatch_list_stall_cycles is not None
                    else 0.0,
                    "RF Stalls": schedule_stage_rf_access_stall_cycles
                    / total_instructions
                    if schedule_stage_rf_access_stall_cycles is not None
                    else 0.0,
                    "LDS Stalls": schedule_stage_lds_stall_cycles
                    / total_instructions
                    if schedule_stage_lds_stall_cycles is not None
                    else 0.0,
                    "Operand Stalls": schedule_stage_operand_stall_cycles
                    / total_instructions
                    if schedule_stage_operand_stall_cycles is not None
                    else 0.0,
                    "Resource Stalls": schedule_stage_resource_stall_cycles
                    / total_instructions
                    if schedule_stage_resource_stall_cycles is not None
                    else 0.0,
                    "Wavefront Stalls": (
                        wavefront_schedule_stage_cycles
                        + wavefront_schedule_stage_stall_cycles
                        + wavefront_schedule_stage_rf_access_stall_cycles
                        + wavefront_schedule_stage_resource_stall_cycles
                        + wavefront_schedule_stage_operand_stall_cycles
                        + wavefront_schedule_stage_lds_stall_cycles
                    )
                    / total_instructions
                    if (
                        wavefront_schedule_stage_cycles is not None
                        and wavefront_schedule_stage_stall_cycles is not None
                        and wavefront_schedule_stage_rf_access_stall_cycles
                        is not None
                        and wavefront_schedule_stage_resource_stall_cycles
                        is not None
                        and wavefront_schedule_stage_operand_stall_cycles
                        is not None
                        and wavefront_schedule_stage_lds_stall_cycles
                        is not None
                    )
                    else 0.0,
                }

                cu_cpi_stack[f"Wavefront {wf_id}"] = {
                    "CPI": cpi,
                    "CPI Stack": cpi_stack,
                }
            else:
                print(
                    f"Skipping CUs{cu_id}.wavefronts{wf_id}. exec_stage_cycles_with_instr={exec_stage_cycles_with_instr}, exec_stage_cycles_no_instr={exec_stage_cycles_no_instr}, total_instructions={total_instructions}"
                )

        cpi_stacks[cu_id] = cu_cpi_stack

    return cpi_stacks


if __name__ == "__main__":
    import sys

    input_file_path = sys.argv[1]
    cpi_stacks = get_cpi_stack(input_file_path)
    print(cpi_stacks)
